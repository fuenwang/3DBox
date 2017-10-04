import os
import sys
sys.path.append('..')
import cv2
import torch
import Dataset
import numpy as np
from pydriver.datasets import kitti
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
    bins = 2
    w = 1
    alpha = 1
    path = '../../Kitti/training'
    kittiData = kitti.KITTIObjectsReader(path)
    #print kittiData.getFrameInfo(1)['calibration'] ['projection_left']
    #sys.exit()
    data = Dataset.ImageDataset(path)
    data = Dataset.BatchDataset(data, 1, bins, mode='eval')
    #print 'a'    
    param = torch.load('model.pkl')
    VGG = vgg.vgg19_bn(pretrained=False)
    model = Model(features=VGG.features, bins=bins).cuda()
    model.load_state_dict(param)
    model.eval()

    total = 0

    angle_error = []
    dim_error = []
    for epoch in range(1):
        for i in range(5000):
            batch, centerAngle, info = data.EvalBatch()
            P = kittiData.getFrameInfo(info['Index'])['calibration'] ['projection_left']
            box_2D = info['Box_2D']
            print P, box_2D
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()
            [orient, conf, dim] = model(batch)
            

            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            
            #img = Batch2Image(batch)
            theta = np.arctan2(sin, cos) / np.pi * 180
            angle = info['LocalAngle'] / np.pi * 180
            theta =  theta + centerAngle[argmax] / np.pi * 180            
            if theta < 0:
                theta += 360
            error = abs(angle - theta)
            if error > 180:
                error = 360 - error
            angle_error.append(abs(error))

            dimGT = info['Dimension']
            norm = np.sum(np.abs(dimGT - dim)) / 3
            dim_error.append(norm)
            total += 1

            sys.exit()
            if i % 40 == 0:
                print '===='
                print info['Ry']
                print info['ThetaRay']
                print 'GT angle: %ld'%(angle)
                print 'Predict angle: %ld'%theta
                print 'GT dim: ', dimGT
                print 'Predict dim: ', dim
                print '===='

    angle_error = np.array(angle_error)
    dim_error = np.array(dim_error)
    print np.mean(angle_error)
    print np.mean(dim_error)
    np.save('angle_error.npy', angle_error)
    np.save('dim_error.npy', dim_error)
