import os
import sys
import cv2
import torch
import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg
from Model import Model
from torch.autograd import Variable


def Batch2Image(batch):
    img = np.zeros([224, 224, 3], np.float32)
    try:
        batch = batch.cpu().data.numpy()
    except:
        pass

    img[:, :, 0] = batch[0, 2, :, :]
    img[:, :, 1] = batch[0, 1, :, :]
    img[:, :, 2] = batch[0, 0, :, :]
    return img

if __name__ == '__main__':
    bins = 8
    w = 1
    alpha = 1
    data = Dataset.ImageDataset('../Kitti/training')
    data = Dataset.BatchDataset(data, 1, bins, mode='eval')
    #print 'a'    
    param = torch.load('exp1/model_confidence_orient_dimension_8bin.pkl')
    VGG = vgg.vgg19_bn(pretrained=False)
    model = Model(features=VGG.features, bins=bins).cuda()
    model.load_state_dict(param)
    model.eval()

    total = 0
    right = 0

    angle_error = []
    dim_error = []
    for epoch in range(1):
        for i in range(5000):
            #print '1'
            #for j in range(20):
            batch, confidence, confidence_multi, ntheta, angleDiff, dimGT, angle, Ry, ThetaRay = data.Next()
            #print '2'
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()
            confidence = Variable(torch.FloatTensor(confidence), requires_grad=False).cuda()
            ntheta = Variable(torch.FloatTensor(ntheta), requires_grad=False).cuda() 
            angleDiff = Variable(torch.FloatTensor(angleDiff), requires_grad=False).cuda()
            #dimGT = Variable(torch.FloatTensor(dimGT), requires_grad=False).cuda()
            [orient, conf, dim] = model(batch)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            
            img = Batch2Image(batch)
            theta = np.arctan2(sin, cos) / np.pi * 180
            confidence = confidence.cpu().data.numpy()[0, :]
            if confidence[np.argmax(conf)] == 1:
                right += 1
            total += 1
            angle = angle / np.pi * 180
            theta =  theta + data.centerAngle[argmax] / np.pi * 180            
            error = abs(angle - theta)
            if error > 180:
                error = 360 - error
            angle_error.append(abs(error))
            dimGT = dimGT[0, :]
            norm = np.sum(np.abs(dimGT - dim)) / 3
            dim_error.append(norm)
            if theta < 0:
                theta += 360
            if i % 40 == 0:
                print '===='
                print Ry
                print ThetaRay
                print 'Class: %ld %%'%(float(right) / total * 100)
                print 'GT angle: %ld'%(angle)
                print 'Predict angle: %ld'%theta
                print 'GT dim: ', dimGT
                print 'Predict dim: ', dim
                print '===='
            #cv2.namedWindow('GG')
            #cv2.imshow('GG', img)
            #cv2.waitKey(0)

    #print 'Avg angle error: %lf'%(a / 5000)
    #print 'Avg distance: %lf' %(b / 5000)   
    angle_error = np.array(angle_error)
    dim_error = np.array(dim_error)
    print np.mean(angle_error)
    print np.mean(dim_error)
    np.save('exp1/angle_error.npy', angle_error)
    np.save('exp1/dim_error.npy', dim_error)
