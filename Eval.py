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

def Error(x, P, homo_vertex, x_bound, y_bound, tmp):
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
    error = np.abs([min_x - x_bound[0], max_x - x_bound[1], min_y - y_bound[0], max_y - y_bound[1]])
    #error = np.abs([0, max_x - x_bound[1], min_y - y_bound[0], max_y - y_bound[1]])
    return error



def GetTranslation(P, box, orientation, dimension, init=None):
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
        x0 = np.array([-0.72, 1.9, 76.1], np.float)
    else:
        x0 = init
    bound = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
    result = least_squares(Error, x0, verbose=2, bounds=bound, args=(P, homo_vertex, x_bound, y_bound, tmp))
    return result['x']


if __name__ == '__main__':
    '''
    P = np.array(
            [[  7.18335100e+02,   0.00000000e+00,   6.00389100e+02,   4.45038200e+01],
             [  0.00000000e+00,   7.18335100e+02,   1.81512200e+02,  -5.95110700e-01],
             [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   2.61631500e-03]]
        , np.float)
    box_2D = [(548, 171), (572, 194)]
    orient = -92.8191628112
    dim = np.array([1.48, 1.56, 3.62])
    GT = [-2.72, 0.82, 48.22]
    print dim
    print GetTranslation(P, box_2D, orient, dim, init = np.array(GT))
    print GT
    '''
    data = Dataset.ImageDataset('../Kitti/training/')
    data = Dataset.BatchDataset(data, mode = 'eval')
    kitti =  pydriver.datasets.kitti.KITTIObjectsReader('../Kitti/training') 
    
    #data.idx = 35611  #problem !!!!!!!!!!!!!!!
    data.idx = 25
    batch, centerAngle, info = data.EvalBatch()
    print info
    tmp = kitti.getFrameInfo(info['Index'])['calibration']
    #print tmp['projection_left']
    #print tmp['reprojection']
    #print tmp.keys()
    P = tmp['projection_left']
    print P.astype(int)
    #sys.exit()
    box_2D = info['Box_2D'] 
    orient = info['Ry']
    dim = info['Dimension']
    GT = info['Location']
    GT[1] -= 0.5 * dim[0]
    print 'orient:', orient
    print 'box: ', box_2D
    print 'dim: ', dim
    print 'GT: ', GT
    print '=============='
    #print GetTranslation(P, box_2D, orient, np.array(dim), init = np.array(GT))
    print GetTranslation(P, box_2D, orient, np.array(dim))






