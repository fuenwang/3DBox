import numpy as np
import matplotlib.pyplot as plt


angle = np.load('angle_error.npy')
dim = np.load('dim_error.npy')



print np.mean(angle)
print np.mean(dim)
