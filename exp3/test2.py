import os
import sys
sys.path.append('..')
import cv2
import Eval
import torch
import Dataset
import numpy as np
import pydriver
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
    bins = 8
    w = 1
    alpha = 1
    path = '../../Kitti/training'
    kittiData = kitti.KITTIObjectsReader(path)
    #print kittiData.getFrameInfo(0)['calibration']
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
    error_lst = []
    distance_lst = []
    for epoch in range(1):
        for i in range(5000):
            #data.idx = 10
            batch, centerAngle, info = data.EvalBatch()
            P = kittiData.getFrameInfo(info['Index'])['calibration'] ['projection_left']
            box_2D = info['Box_2D']
            dimGT = info['Dimension']
            angle = info['LocalAngle'] / np.pi * 180
            #print P, box_2D, dimGT
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
            theta =  theta + centerAngle[argmax] / np.pi * 180
             
            orientation_estimate = pydriver.common.functions.pyNormalizeAngle(np.radians(360 - info['ThetaRay'] - theta))
            orientation_estimate = orientation_estimate / np.pi * 180
            error = abs(orientation_estimate - info['Ry'])
            if error > 180:
                error = abs(360 - error)
            error_lst.append(error)

            Translation = Eval.GetTranslation(P, box_2D, orientation_estimate, dim)
            distance_lst.append(np.linalg.norm(Translation - info['Location']))
            #Translation = Eval.GetTranslation(P, box_2D, info['Ry'], np.array(dimGT))
            #print box_2D, info['Ry'], dimGT
            #print Translation
            #print info['ID']
            #print info['Location']
            #print error
            #print dimGT
            #print dim
            #sys.exit()
            #error = abs(theta - info['LocalAngle'])
            if i % 40 == 0:
                print '===='
                print info['Ry']
                print info['ThetaRay']
                print 'GT Ry: %ld'%(info['Ry'])
                print 'Predict Ry: %ld'%(orientation_estimate)
                print 'GT dim: ', dimGT
                print 'Predict dim: ', dim
                print 'Distance: %lf'%np.linalg.norm(Translation - info['Location'])
                print '===='
    print 'Mean error: ', np.mean(error_lst)
    print 'Std: ', np.std(error_lst)
    print 'Mean error', np.mean(distance_lst)
    print 'Std: ', np.std(distance_lst)
