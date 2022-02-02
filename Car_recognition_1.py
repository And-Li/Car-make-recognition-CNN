import os
from PIL import Image
import random
import matplotlib.pyplot as plt


IMAGE_PATH = 'middle_fmr'
print(os.listdir(IMAGE_PATH))  # just to be sure you've mounted the right directory

CLASS_LIST = sorted(os.listdir(IMAGE_PATH))  # we'll sort the dirs, 'cause .listdir returns content randomly
CLASS_COUNT = len(CLASS_LIST)  # we get the quantity of the classes
print(f'Class quantity is {CLASS_COUNT}, class labels are {CLASS_LIST}')  # checking if everything is right

for cls in CLASS_LIST:
    print(cls, ":", os.listdir(f'{IMAGE_PATH}/{cls}/'))