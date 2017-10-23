import os
import sys
import cv2
import Dataset
import numpy as np
from scipy.optimize import least_squares
import pydriver
from pydriver.common.functions import  pyNormalizeAngle

def CreateVertex(dimension):
    [dy, dz, dx] = 0.5 * dimension
    #[dz, dy, dx] = 0.5 * dimension
    vertex = [
                [dx, dy, dz],
                [dx, dy, -dz],
                [dx, -dy, dz],
                [dx, -dy, -dz],
                [-dx, dy, dz],
                [-dx, dy, -dz],
                [-dx, -dy, dz],
                [-dx, -dy, -dz]
            ]
    return np.array(vertex, np.float)

def Error(x, P, homo_vertex, x_bound, y_bound, tmp, xedge, yedge):
    tmp[:, :] = homo_vertex[:, :]
    tmp[:, 0] += x[0]
    tmp[:, 1] += x[1]
    tmp[:, 2] += x[2]
    
    #print tmp
    reproject = np.dot(P, tmp.T).T 
    #print reproject
    reproject[:, 0] /= reproject[:, 2]
    reproject[:, 1] /= reproject[:, 2]
    reproject[:, 2] = 1
    #print reproject
    min_x = np.min(reproject[:, 0])
    max_x = np.max(reproject[:, 0])
    min_y = np.min(reproject[:, 1])
    max_y = np.max(reproject[:, 1])

    #error = np.zeros(4, np.float)
    '''
    if xedge == 0:
        xerror = np.abs([min_x - x_bound[0], max_x - x_bound[1]])
    elif xedge == 1: # left edge
        xerror = np.abs([0, max_x - x_bound[1]])
    elif xedge == 2: # right edge
        xerror = np.abs([min_x - x_bound[0], 0])
    if yedge == 0:
        yerror = np.abs([min_y - y_bound[0], max_y - y_bound[1]])
    elif yedge == 1: # top edge
        yerror = np.abs([0, max_y - y_bound[1]])
    elif yedge == 2: # bottom edge
        yerror = np.abs([min_y - y_bound[0], 0])
    '''
    factor = 0
    factor2 = 1
    if xedge == 0:
        xerror = np.abs([min_x - x_bound[0], max_x - x_bound[1]])
    elif xedge == 1: # left edge 
        xerror = np.abs([factor * (min_x - x_bound[0]), factor2 * (max_x - x_bound[1])])
    elif xedge == 2: # right edge
        xerror = np.abs([factor2 * (min_x - x_bound[0]), factor * (max_x - x_bound[1])])

    if yedge == 0:
        yerror = np.abs([min_y - y_bound[0], max_y - y_bound[1]])
    elif yedge == 1: # top edge
        yerror = np.abs([factor * (min_y - y_bound[0]), factor2 * (max_y - y_bound[1])])
    elif yedge == 2: # bottom edge
        yerror = np.abs([factor2 * (min_y - y_bound[0]), factor * (max_y - y_bound[1])])

    error = np.hstack([xerror, yerror])
    return error



def GetTranslation(P, box, orientation, dimension, init=None, imgSize = (375, 1242), verbose = 0):
    #[height, width, length] = dimention
    #[dy, dz, dx] = dimention
    pt1 = box[0]
    pt2 = box[1]
    x_bound = [pt1[0], pt2[0]]
    y_bound = [pt1[1], pt2[1]]

    vertex = CreateVertex(dimension)
    R = cv2.Rodrigues(np.array([0, np.radians(orientation), 0]))[0]
    vertex = np.dot(R, vertex.T).T # After rotation
    #print vertex
    tmp = np.zeros([8, 4], np.float)
    homo_vertex = np.ones([8, 4], np.float)
    homo_vertex[:, :3] = vertex[:, :]
    #print Error(init, P, homo_vertex, x_bound, y_bound, tmp)
    #sys.exit()
    
    #x0 = np.zeros(3) + 5
    if init is None:
        x0 = np.array([5, 1, 30], np.float)
        x1 = np.array([-5, 1, 30], np.float)
    else:
        x0 = init
        x1 = init

    if x_bound[0] == 0:
        xedge = 1
    elif x_bound[1] == imgSize[1] - 1:
        xedge = 2
    else:
        xedge = 0
    if y_bound[0] == 0:
        yedge = 1
    elif y_bound[1] == imgSize[0] - 1:
        yedge = 2
    else:
        yedge = 0
    #yedge = 0
    #[xedge, yedge] = [0, 0]
    bound = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
    result = least_squares(Error, x0, verbose=verbose, bounds=bound, method='trf',
            args=(P, homo_vertex, x_bound, y_bound, tmp, xedge, yedge))
    x = result['x']
    '''
    if result['cost'] > 1e-1:
        result2 =  least_squares(Error, x1, verbose=0, bounds=bound, method='trf',
            args=(P, homo_vertex, x_bound, y_bound, tmp, xedge, yedge))
        if result2['cost'] < result['cost']:
            x = result2['x']
    '''
    x[1] += 0.5 * dimension[0]
    #print (xedge, yedge)
    return x


if __name__ == '__main__':
    data = Dataset.ImageDataset('../Kitti/training/')
    data = Dataset.BatchDataset(data, mode = 'eval')
    kitti =  pydriver.datasets.kitti.KITTIObjectsReader('../Kitti/training') 
    
    #data.idx = 35611  #problem !!!!!!!!!!!!!!!
    lst = []
    for i in range(5000):
        #data.idx = 35612
        #data.idx = 35611
        batch, centerAngle, info = data.EvalBatch()
        tmp = kitti.getFrameInfo(info['Index'])
        #print tmp
        #exit()
        #print tmp['labels'][0]
        #print tmp['projection_left']
        #print tmp['reprojection']
        #print tmp.keys()
        P = tmp['calibration']['projection_left']
        #print P
        #print P.astype(int)
        #sys.exit()
        box_2D = info['Box_2D'] 
        orient = info['Ry']
        dim = info['Dimension']
        GT = info['Location']
        predict =  GetTranslation(P, box_2D, orient, np.array(dim), init=np.array(GT), verbose=0)
        #predict =  GetTranslation(P, box_2D, orient, np.array(dim), verbose=0)
        error = np.linalg.norm(predict - np.array(GT))
        lst.append(error)
        #GT[1] -= 0.5 * dim[0]
        if i % 10 != 0:
            continue
        print '=============='
        print data.idx - 1
        print 'orient:', orient
        print 'box: ', box_2D
        print 'dim: ', dim
        print 'GT: ', GT
        print 'Predict: ',predict
        print 'Distance: %lf'%np.linalg.norm(predict - np.array(GT))
        print '=============='
        #print GetTranslation(P, box_2D, orient, np.array(dim), init = np.array(GT))

    print np.mean(lst)




