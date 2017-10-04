import os
import sys
import cv2
import numpy as np
from pydriver.common.functions import  pyNormalizeAngle

def GetTranslation(P, box, dimention):
    #[height, width, length] = dimention
    [dy, dz, dx] = dimention
    pt1 = box[0]
    pt2 = box[1]
    x = [pt1[0], pt2[0]]
    y = [pt1[1], pt2[1]]
