import os
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from pydriver.common.functions import  pyNormalizeAngle

def CreateVertex(dimension):
    [dy, dz, dx] = 0.5 * dimension
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

    reproject = np.dot(P, tmp.T).T
    reproject[:, 0] /= reproject[:, 2]
    reproject[:, 1] /= reproject[:, 2]
    min_x = np.min(reproject[:, 0])
    max_x = np.max(reproject[:, 0])
    min_y = np.min(reproject[:, 1])
    max_y = np.max(reproject[:, 1])

    #error = np.zeros(4, np.float)
    error = np.abs([min_x - x_bound[0], max_x - x_bound[1], min_y - y_bound[0], max_y - y_bound[1]])
    return error



def GetTranslation(P, box, orientation, dimension):
    #[height, width, length] = dimention
    #[dy, dz, dx] = dimention
    pt1 = box[0]
    pt2 = box[1]
    x_bound = [pt1[0], pt2[0]]
    y_bound = [pt1[1], pt2[1]]

    vertex = CreateVertex(dimension)
    R = cv2.Rodrigues(np.array([0, np.radians(orientation), 0]))[0]
    #print R
    #print vertex
    vertex = np.dot(R, vertex.T).T # After rotation
    tmp = np.zeros([8, 4], np.float)
    homo_vertex = np.ones([8, 4], np.float)
    homo_vertex[:, :3] = vertex[:, :]
    
    #x0 = np.zeros(3) + 5
    x0 = np.array([-0.72, 1.9, 76.1], np.float)
    result = least_squares(Error, x0, verbose=2, args=(P, homo_vertex, x_bound, y_bound, tmp))
    return result['x']

