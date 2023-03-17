import hdf5storage
import cv2
import numpy as np
import os, cv2
from networks_gssl import *
import data_utils_gssl
from functions_gssl import * 



x = [1,2,3]
b = ['a','b','c']
plt.plot(x,b)
if not os.path.exists('./graphs'):
    os.mkdir('./graphs')
graphfilename = os.path.join('./graphs',"%s_%d_epoch_erorr.png" % ('name',15))
plt.savefig(graphfilename)