import os
import sys
sys.path.append('..')
import cv2
import torch
import Dataset
import Data
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
    bins = 2
    w = 1
    alpha = 1
    dataset = Dataset.ImageDataset('../../Kitti/training')
    dataset = Dataset.BatchDataset(dataset, 1, bins, mode='eval')
    #print 'a'    
    param = torch.load('model.pkl')
    VGG = vgg.vgg19_bn(pretrained=False)
    model = Model(features=VGG.features, bins=bins).cuda()
    model.load_state_dict(param)
    model.eval()

    ID = '000012'
    data = Data.Data(ID)

    stop = False
    while not stop:
        [batch, stop, crop, info] = data.Next()
        batch_v = Variable(torch.FloatTensor(batch), requires_grad = False).cuda()

        [orient, confi, dim] = model(batch_v)
        confi = F.softmax(confi)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = confi.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]
        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        conf = conf[argmax]
        cos = orient[0]
        sin = orient[1]
        theta = np.arctan2(sin, cos) / np.pi * 180
        theta =  theta + dataset.centerAngle[argmax] / np.pi * 180 
        if theta < 0:
            theta += 360
        data.Update(info, theta, dim.tolist(), conf)
        print theta
        print dim
        print conf
        #cv2.namedWindow('GG')
        #cv2.imshow('GG', crop)
        #cv2.waitKey(0)
    data.WriteFile('data/%s.txt'%ID)
