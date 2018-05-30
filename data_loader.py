"""
@author: Md Mostafa Kamal Sarker
@ Department of Computer Engineering and Mathematics, Universitat Rovira i Virgili, 43007 Tarragona, Spain
@ email: m.kamal.sarker@gmail.com
@ Date: 23.05.2017
"""
import os
from os import listdir
from os.path import isfile, join
import collections
from collections import defaultdict
import cv2
import numpy as np

###################### Get the class name from the directory  ######################################
src_dir='/media/mostafa/Data/Food/FoodRelated_DATASET/data/train'  
class_name=os.listdir(src_dir)
print (class_name)
nub_class=len(class_name)
print (nub_class)
######################  maps from class to index and vice vers ########################################
class_to_ix = {}
ix_to_class = {}
class_to_ix = dict(zip(class_name, range(len(class_name))))
ix_to_class = dict(zip(range(len(class_name)), class_name))
class_to_ix = {v: k for k, v in ix_to_class.items()}
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

############ Load dataset images and resize to meet minimum width and height pixel size ############
def load_images(root):
    all_imgs = []
    all_classes = []
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        for img_name in imgs:
            img_arr = cv2.imread(join(root, subdir, img_name))
            img_arr = cv2.resize(img_arr, (299, 299)) ## change the size here with model input
            img_arr_rs = img_arr
            all_imgs.append(img_arr_rs)
            all_classes.append(class_ix)
    print(len(all_imgs), 'images loaded')
    return np.array(all_imgs), np.array(all_classes)

#######################################################################################################
