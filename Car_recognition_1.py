import os
from PIL import Image
import random
import matplotlib.pyplot as plt


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

