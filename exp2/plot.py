import numpy as np
import matplotlib.pyplot as plt

angle = np.load('angle_error_train.npy')
angle_mean = np.mean(angle)
angle_std = np.std(angle)
print angle_mean
print angle_std
dim = np.load('dim_error_train.npy')
dim_mean = np.mean(dim)
dim_std = np.std(dim)
print dim_mean
print dim_std
#'''
plt.subplot('111')
x_axis = np.arange(5000) + 1
plt.plot(x_axis, angle, 'oc', label='angle error', markersize=1)
plt.plot(x_axis, np.zeros(5000)+angle_mean, 'r', label='mean')
plt.legend()
plt.title('Angle Error(avg = 4.4, std = 8.4)')
'''
plt.subplot('111')
x_axis = np.arange(5000) + 1
plt.plot(x_axis, dim, 'oc', label='dim error', markersize=1)
plt.plot(x_axis, np.zeros(5000)+dim_mean, 'r', label='mean')
plt.legend()
plt.title('Dimension Error(avg = 0.21, std = 0.12)')
'''
plt.savefig('angle_train.png', dpi=300, bbox_inches='tight')
#plt.show()



