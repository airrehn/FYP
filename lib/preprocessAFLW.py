import os, cv2
import hdf5storage
import numpy as np
import sys

def gen_meanface(root_folder, data_name):
    with open(os.path.join(root_folder, data_name, 'train.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split()[1:] for x in annos]
    annos = [[float(x) for x in anno] for anno in annos]
    annos = np.array(annos)
    meanface = np.mean(annos, axis=0) #divides mean of columns
    meanface = meanface.tolist()
    meanface = [str(x) for x in meanface]
    
    with open(os.path.join(root_folder, data_name, 'meanface.txt'), 'w') as f:
        f.write(' '.join(meanface))


def process_aflw(root_folder, image_name, anno, target_size):
    image = cv2.imread(os.path.join(root_folder, 'nAFLW', 'images', image_name))
    image_height, image_width, _ = image.shape
    anno_x = anno[0::2]
    anno_y = anno[1::2]

    #just safety checks
    anno_x = [x if x >=0 else 0 for x in anno_x] 
    anno_x = [x if x <=image_width else image_width for x in anno_x]
    anno_y = [y if y >=0 else 0 for y in anno_y] 
    anno_y = [y if y <=image_height else image_height for y in anno_y] 

    
    xmin = 0 #l
    ymin = 0 #t
    xmax = image_width-1 #r
    ymax = image_height-1 #b

    #image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]

    image_crop = cv2.resize(image, (target_size, target_size)) #resize to 256x256

    anno_x = (np.array(anno_x) - xmin) / (xmax - xmin) #x coordinates becomes a smaller number from 0 ~ 1
    anno_y = (np.array(anno_y) - ymin) / (ymax - ymin) #y coordinates become a smaller number from 0 ~ 1

    anno = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1).flatten()
    anno = zip(anno[0::2], anno[1::2])
    return image_crop, anno



def gen_data(root_folder, target_size):
    data_name = 'nAFLW'
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))
    ################################################################################################################


    nameList,annos = np.array_split(np.loadtxt(os.path.join('../data','nAFLW', 'EDITTED_labels.txt'), dtype = str), [1], axis=1)
    nameList = nameList.flatten()
    annos = annos.astype(float)
    

    with open(os.path.join(root_folder, 'nAFLW', 'train.txt'), 'w') as f:
        for index in range(0,1600):
            # from matlab index
            image_name = nameList[index] #index-1 because matlab range is 1~x, whereas index range is 0~x-1

            anno = annos[index]
            image_crop, anno = process_aflw(root_folder, image_name, anno, target_size) #gets image files and annotation
            pad_num = 5-len(str(index+1))
            image_crop_name = 'aflw_train_' + '0' * pad_num + str(index+1) + '.jpg'
            print(image_crop_name)
            cv2.imwrite(os.path.join(root_folder, 'nAFLW', 'images_train', image_crop_name), image_crop) #saves image to aflw/images_train
            f.write(image_crop_name+' ') #writes into train.txt
            for x,y in anno:
                f.write(str(x)+' '+str(y)+' ')
            f.write('\n')

    with open(os.path.join(root_folder, 'nAFLW', 'test.txt'), 'w') as f:
        for index in range(1600,2000):
            # from matlab index
            image_name = nameList[index]

            anno = annos[index]
            image_crop, anno = process_aflw(root_folder, image_name, anno, target_size)
            pad_num = 5-len(str(index))
            image_crop_name = 'aflw_test_' + '0' * pad_num + str(index+1) + '.jpg'
            print(image_crop_name)
            cv2.imwrite(os.path.join(root_folder, 'nAFLW', 'images_test', image_crop_name), image_crop)
            f.write(image_crop_name+' ')
            for x,y in anno:
                f.write(str(x)+' '+str(y)+' ')
            f.write('\n')

    gen_meanface(root_folder, data_name) #gets meanface.txt basically mean value of annotations (each colum) 



if __name__ == '__main__':
    gen_data('../data', 256)