import cv2
import numpy as np
import os
import skimage
import time
import pickle

def by_frame_num(name):
    if len(name) == 11: num = name[6]
    elif len(name) == 12: num = name[6:8]
    elif len(name) == 13: num = name[6:9]
    elif len(name) == 14: num = 1000
    try: 
        return int(num)
    except: print(name)

def by_int(name):
    return int(name)


def import_images(path):
    set_list = os.listdir(path)
    set_list.remove('.DS_Store')
    set_list = sorted(set_list, key=by_int)

    all_images = []
    for image_set in set_list:
        images_list = os.listdir(path+'/'+image_set)
        if '.DS_Store' in images_list: images_list.remove('.DS_Store')
        images_set = []
        for img in sorted(images_list, key=by_frame_num):
            img_array = cv2.imread(path+'/'+image_set+'/'+img, 0)
            img_array = img_array/255
            zeros = np.where(img_array==0)
            ones = np.where(img_array==1)
            img_array[ones] = 0
            img_array[zeros] = 1
            images_set.append(img_array)
        images_set = np.asarray(images_set)
        all_images.append(images_set)
    all_images = np.asarray(all_images)
    return all_images 

start = time.time()

images = import_images('64_images')
with open('images_data.txt', 'wb') as fp:
    pickle.dump(images, fp)
fp.close()

def grab_images(path):
    with open(path, 'rb') as fp:
        images = pickle.load(fp)
    fp.close()
    return images

