import hdf5storage
import cv2
import numpy as np
import os, cv2
from networks_gssl import *
import data_utils_gssl
from functions_gssl import * 



nameList = np.loadtxt(os.path.join('../data','CELEBA', 'train.txt'), dtype = str)
#nameList = nameList.flatten()
for name in nameList:

    print(name)

#labels = get_label('hoof', 'train_300W.txt')
#print(labels)


"""
mat = hdf5storage.loadmat('../data/AFLW/AFLWinfo_release.mat')

bboxes = mat['bbox']
annos = mat['data']
mask_new = mat['mask_new']
nameList = mat['nameList']
ra = mat['ra'][0]
train_indices = ra[:20000]

print(nameList[0][0][0])
print(annos[0])


image_name = nameList[0][0][0]
#########################################################################
#print(annos)
#print(nameList[0][0][0])


testanno = annos[7138-1] #corresponds to image00014

anno_x = testanno[:19]
anno_y = testanno[19:]

image = cv2.imread('D:/project/PIPNet/data/nAFLW/images/image00014.jpg')
image_height,image_width,_ = image.shape

anno_x = [x if x >=0 else 0 for x in anno_x] 
anno_x = [x if x <=image_width else image_width for x in anno_x]
anno_y = [y if y >=0 else 0 for y in anno_y] 
anno_y = [y if y <=image_height else image_height for y in anno_y] 

xmin = 0 #l
ymin = 0 #t
xmax = image_width-1 #r
ymax = image_height-1 #b

#anno_x = (np.array(anno_x) - xmin) / (xmax - xmin) #x coordinates becomes a smaller number from 0 ~ 1
#anno_y = (np.array(anno_y) - ymin) / (ymax - ymin)

anno_x = np.array(anno_x)
anno_y = np.array(anno_y)
n = anno_x.reshape(-1,1) # 1 column of annox's

m = anno_y.reshape(-1,1)

#z = np.concatenate([n, m], axis=1)

z = np.concatenate([n, m], axis=1).flatten() #flatten puts all in 1 array, x1y1 x2y2 x3y3 etc

print('\n\n', n, '\n\n', m, '\n\n', z)


z2 = zip(z[0::2], z[1::2])

#print('\n\n', n, '\n\n', m, '\n\n', z)



print(z[0::2])


annos= np.array([])
nameList = np.array([])

nameList,annos = np.array_split(np.loadtxt(os.path.join('../data','nAFLW', 'EDITTED_labels.txt'), dtype = str), [1], axis=1)
nameList = nameList.flatten()
annos = annos.astype(float)
print (nameList[0])
print(annos[0])

annosx = annos[0][0::2]
annosy = annos[0][1::2]

print(annosx)
print(annosy)

n = annosx.reshape(-1,1) # 1 column of annox's

m = annosy.reshape(-1,1)

z = np.concatenate([n, m], axis=1).flatten() 

with open(os.path.join('../data','nAFLW', 'EDITTED_labels.txt'),'r') as file:
    lines = file.readlines()
    for line in lines:
        row = line.split()
        nameList.append(row[0])
        annos.append(row[1:])






image = cv2.imread('D:/project/PIPNet/data/nAFLW/images/image00014.jpg')
imageh,imagew,_ = image.shape
print(imageh-1)
"""