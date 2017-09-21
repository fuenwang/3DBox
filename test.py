import os
import sys
import cv2
import torch
import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    bins = 3
    w = 1
    alpha = 1
    data = Dataset.ImageDataset('../Kitti/training')
    data = Dataset.BatchDataset(data, 1, bins)
    #print 'a'    
    model = torch.load('model.pkl').cuda()
    #print 'b'
    #model = Model.Model()
    for epoch in range(1):
        for i in range(1):
            #print '1'
            for j in range(1):
                batch, confidence, ntheta, angleDiff, dimGT, angle = data.Next()
            #print '2'
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()
            confidence = Variable(torch.FloatTensor(confidence), requires_grad=False).cuda()
            ntheta = Variable(torch.FloatTensor(ntheta), requires_grad=False).cuda() 
            angleDiff = Variable(torch.FloatTensor(angleDiff), requires_grad=False).cuda()
            dimGT = Variable(torch.FloatTensor(dimGT), requires_grad=False).cuda()
            #print 'SSS'         
            [orient, conf, dim] = model(batch)
            #print 'GG'
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            #print orient
            print conf
            print confidence
            print angle / np.pi * 180
            #exit()
            argmax = np.argmax(conf)
            
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            
            img = Batch2Image(batch)
            theta = np.arctan2(sin, cos) / np.pi * 180
            print theta
            cv2.namedWindow('GG')
            cv2.imshow('GG', img)
            cv2.waitKey(0)




