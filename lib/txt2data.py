import os, cv2
import numpy as np
import sys

with open(('D:/project/aflw/neckAFLW_labels.txt'), 'r') as f, open('D:/project/aflw/EDITTED_labels.txt', 'w') as newf:
    lines = f.readlines()
    for line in lines:
        s = line.replace('D:/Dataset/AFLW/part-of-AFLW/already_labeled/','')
        newf.write(s)
    