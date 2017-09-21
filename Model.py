import os
import sys
import cv2
import torch
from torchvision.models import vgg
import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#def OrientationLoss(orient, orientGT)
def OrientationLoss(orient, angleDiff, confGT):
    #
    # orid = [sin(delta), cos(delta)] shape = [batch, bins, 2]
    # angleDiff = GT - center, shape = [batch, bins]
    #
    [batch, _, bins] = orient.size()
    cos_diff = torch.cos(angleDiff)
    sin_diff = torch.sin(angleDiff)
    cos_ori = orient[:, :, 0]
    sin_ori = orient[:, :, 1]
    mask1 = (confGT != 0)
    mask2 = (confGT == 0)
    count = torch.sum(mask1, dim=1)
    #print cos_ori
    #print sin_ori
    #print orient
    #print cos_diff
    #print cos_ori
    #print count
    tmp = cos_diff * cos_ori + sin_diff * sin_ori
    #print tmp
    tmp[mask2] = 0
    total = torch.sum(tmp, dim = 1)
    #print total
    count = count.type(torch.FloatTensor).cuda()
    #print count
    #print total
    total = total / count
    #print total.requires_grad
    return torch.sum(total) / batch
    #print total
    #exit()

class Model(nn.Module):
    def __init__(self, features, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        '''
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        '''
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins)
                    #nn.Softmax()
                    #nn.Sigmoid()
                )
        '''
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )
        '''

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        #orientation = self.orientation(x)
        #orientation = orientation.view(-1, self.bins, 2)
        #orientation = F.normalize(orientation, dim=1)
        confidence = self.confidence(x)
        #dimension = self.dimension(x)
        return None, confidence, None



if __name__ == '__main__':
    bins = 8
    w = 20
    alpha = 1
    data = Dataset.ImageDataset('../Kitti/training')
    data = Dataset.BatchDataset(data, 8, bins)
    
    #vgg = torch.load('model/vgg16.pkl').cuda()
    vgg = vgg.vgg19_bn(pretrained=True) 
    model = Model(features = vgg.features, bins=bins).cuda()
    #model = torch.load('model.pkl').cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    #conf_LossFunc = nn.MSELoss().cuda()
    #conf_LossFunc = nn.MultiLabelMarginLoss()
    conf_LossFunc = nn.CrossEntropyLoss()
    #conf_LossFunc = nn.NLLLoss().cuda()
    for epoch in range(25):
        for i in range(2000):
            batch, confidence, ntheta, angleDiff, dimGT, LocalAngle, Ry, ThetaRay = data.Next()
            #print confidence
            #continue
            #exit()
            #print confidence
            #exit()
            #print confidence
            #print np.argmax(confidence, axis = 1)
            #exit()
            confidence_arg = np.argmax(confidence, axis = 1)
            #confidence_arg[:] = 2
            #print dimGT
            #exit()
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()
            confidence = Variable(torch.LongTensor(confidence.astype(np.int)), requires_grad=False).cuda()
            ntheta = Variable(torch.FloatTensor(ntheta), requires_grad=False).cuda() 
            angleDiff = Variable(torch.FloatTensor(angleDiff), requires_grad=False).cuda()
            dimGT = Variable(torch.FloatTensor(dimGT), requires_grad=False).cuda()
            confidence_arg = Variable(torch.LongTensor(confidence_arg.astype(np.int)), requires_grad=False).cuda()
            [orient, conf, dim] = model(batch)
            #print confidence_arg
            #print confidence
            #print conf
            #exit()

            #print confidence
            #print conf
            conf_loss = conf_LossFunc(conf, confidence_arg)
            #orient_loss = OrientationLoss(orient, angleDiff, confidence)
            #dim_loss = conf_LossFunc(dim, dimGT)
            #loss_theta = conf_loss + w * (1 - orient_loss)
            loss = conf_loss
            if i % 15 == 0:
                print confidence
                print 'loss_conf'
                print conf_loss
                #print 'loss_orient'
                #print (1 - orient_loss)
                #print 'total'
                #print loss
                print '======='
            #loss = alpha * dim_loss + loss_theta
            if i % 150 == 0:
                #print loss
                torch.save(model, 'model.pkl')
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step() 
    torch.save(model, 'model.pkl')







