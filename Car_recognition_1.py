import os
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np


IMAGE_PATH = 'middle_fmr'
print(os.listdir(IMAGE_PATH))  # just to be sure you've mounted the right directory

CLASS_LIST = sorted(os.listdir(IMAGE_PATH))  # we'll sort the dirs, 'cause .listdir returns content randomly
CLASS_COUNT = len(CLASS_LIST)  # we get the quantity of the classes
print(f'Class quantity is {CLASS_COUNT}, class labels are {CLASS_LIST}')  # checking if everything is right

'''for cls in CLASS_LIST:  # we get lists of files of all the classes
    print(cls, ":", os.listdir(f'{IMAGE_PATH}/{cls}/'))'''
# now we make 2 empty lists for the files:
data_files = []  # list of paths to the files
data_labels = []  # list of corresponding labels
for class_label in range(CLASS_COUNT):  # for every class(of 3)
    class_name = CLASS_LIST[class_label]
    class_path = IMAGE_PATH + '/' + class_name  # setting the full path to the dir with imgs
    class_files = os.listdir(class_path)  # getting the list of img file names of the current class
    print(f'Class size of {class_name} is {len(class_files)} cars')
# adding all files of the class to the empty list
    data_files += [f'{class_path}/{file_name}' for file_name in class_files]
# adding all labels of the current class to the empty list
# so that class file index and label file index would correspond
    data_labels += [class_label] * len(class_files)
print(f'Total size of dataset for learning is {len(data_labels)} files')
# let's check it out:
print('Paths are: ', data_files[1085:1090])
print('Their labels are: ', data_labels[1085:1090])
# Time to transform datasets into np.arrays for learning:
IMG_WIDTH = 128  # bringing imgs to one size
IMG_HEIGHT = 64
data_images = []  # empty list for imgs
for file_name in data_files:
    img = Image.open(file_name).resize((IMG_WIDTH, IMG_HEIGHT))  # opening and resizing every img
    img_np = np.array(img)  # into numpy array
    data_images.append(img_np)  # adding to the empty list of images
x_data = np.array(data_images)  # img list to numpy array
y_data = np.array(data_labels)  # labels list to numpy array
# checking if everything is ok:
print(f'We gathered in array {len(data_images)} photos shaped as following: {img_np.shape}')
print(f'The img-dataset-ready-4-learning is shaped like that: {x_data.shape}')
print(f'The label-dataset-ready-4-learning is shaped like that: {y_data.shape}')

# now we bring the data in img-dataset to be in range (0,1),
# 'cause right now they look like following:
# print(x_data[0])
x_data = x_data/255.  # now everything is ready for the learning!



