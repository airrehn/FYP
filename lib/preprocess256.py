import os, cv2
import numpy as np





def process_aflw(root_folder, target_size, filename):
    image = cv2.imread(os.path.join(root_folder, 'nAFLW', 'images', filename))

    image_crop = cv2.resize(image, (target_size, target_size)) #resize to 256x256

    return image_crop



def gen_data(root_folder, target_size):
    data_name = 'nAFLW'
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_256')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_256'))

    pathT = 'D:/project/PIPNet/data/nAFLW/images'

    for filename in os.listdir(pathT): 
        
        print(filename)

        image_crop = process_aflw(root_folder, target_size, filename) #gets image files and annotation

        cv2.imwrite(os.path.join(root_folder, data_name, 'images_256', filename), image_crop) #saves image to nAFLW/images_256






if __name__ == '__main__':
    gen_data('../data', 256)