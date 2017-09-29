import os
import sys
import cv2
sys.path.append('../')
import numpy as np
import Dataset
import Model
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg



if __name__ == '__main__':
    bins = 2
    w = 1
    alpha = 1
    data = Dataset.ImageDataset('../../Kitti/training')
    data = Dataset.BatchDataset(data, 8, bins)
    #'''
    #vgg = torch.load('model/vgg16.pkl').cuda()
    vgg = vgg.vgg19_bn(pretrained=True) 
    #param = torch.load('model.pkl')
    model = Model.Model(features=vgg.features, bins=bins).cuda()
    #model.load_state_dict(param)

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    dim_LossFunc = nn.MSELoss().cuda()
    conf_LossFunc = nn.CrossEntropyLoss().cuda()
    for epoch in range(25):
        for i in range(5000):
            batch, confidence, confidence_multi, ntheta, angleDiff, dimGT, LocalAngle, Ry, ThetaRay = data.Next()
            confidence_arg = np.argmax(confidence, axis = 1)
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()
            confidence = Variable(torch.LongTensor(confidence.astype(np.int)), requires_grad=False).cuda()
            confidence_multi = Variable(torch.LongTensor(confidence_multi.astype(np.int)), requires_grad=False).cuda()
            ntheta = Variable(torch.FloatTensor(ntheta), requires_grad=False).cuda() 
            angleDiff = Variable(torch.FloatTensor(angleDiff), requires_grad=False).cuda()
            dimGT = Variable(torch.FloatTensor(dimGT), requires_grad=False).cuda()
            confidence_arg = Variable(torch.LongTensor(confidence_arg.astype(np.int)), requires_grad=False).cuda()
            [orient, conf, dim] = model(batch)

            conf_loss = conf_LossFunc(conf, confidence_arg)
            orient_loss = Model.OrientationLoss(orient, angleDiff, confidence_multi)
            dim_loss = dim_LossFunc(dim, dimGT)
            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta
            if i % 15 == 0:
                #print confidence
                print 'loss_conf'
                print conf_loss
                print 'loss_orient'
                print orient_loss
                print 'loss_dim'
                print dim_loss
                print 'total'
                print loss
                print '======='
            #loss = alpha * dim_loss + loss_theta
            if i % 150 == 0:
                #print loss
                torch.save(model.state_dict(), 'model.pkl')
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step() 
    torch.save(model.state_dict(), 'model.pkl')


