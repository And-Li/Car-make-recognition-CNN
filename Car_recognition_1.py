import os
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

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
IMG_WIDTH = 128  # for bringing imgs to one size
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

model = Sequential()
# 1st convolutional layer:
model.add(Conv2D(256, (3, 3), name='First_C', padding='same', activation='relu',
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))  # Changed filter quantity (64,128, 3) -> (64, 128, 256)
model.add(BatchNormalization(name='First_B'))  # Remains unchanged (64, 128, 256) -> (64, 128, 256)
# 2nd convolutional layer:
model.add(Conv2D(256, (3, 3), name='Second_C', padding='same', activation='relu',))  # Unchanged (64, 128, 256) -> (64, 128, 256)
model.add(MaxPooling2D(pool_size=(3, 3), name='Second_M'))  # triple compression with some loss;
                                                            # Depth unchanged, 'cause by default padding='valid': (64,128,256) --> (21,42,256)
# 3d convolutional layer:
model.add(Conv2D(256, (3, 3), name='Third_C', padding='same', activation='relu'))  # Unchanged
model.add(BatchNormalization(name='Third_B'))  # Unchanged (21, 42, 256) -> (21, 42, 256)
model.add(Dropout(0.2, name='Third_D'))  # Unchanged (21, 42, 256) -> (21, 42, 256)
# 4th convolutional layer:
model.add(Conv2D(256, (3, 3), name='Fourth_C', padding='same', activation='relu'))  # Unchanged
model.add(MaxPooling2D(pool_size=(3, 3), name='Fourth_M'))  # triple compression (21, 42, 256)->(7, 14, 256)
                                                            # no loss as division is integer (no remainders)
model.add(Dropout(0.2, name='Fourth_D'))  # sizes & depth remain unchanged
# 5th convolutional layer:
model.add(Conv2D(512, (3, 3), name='Fifth_C', padding='same', activation='relu'))  # depth (filter quantity) changed to 512
model.add(BatchNormalization(name='Fifth_B'))  # remains unchanged
# 6th convolutional layer:
model.add(Conv2D(1024, (3, 3), name='Sixth_C', padding='same', activation='relu'))  # depth (filter quantity) changed to 1024 (7, 14, 512)->(7, 14, 1024)
model.add(MaxPooling2D(pool_size=(3, 3), name='Sixth_M'))   # triple compression, with loss (as remainders r left after division)
                                                            # (7, 14, 1024)-->(2, 4, 1024)
model.add(Dropout(0.2, name='Sixth_D'))  # unchanged
# Classification layer:
model.add(Flatten(name='Class_1'))  # all dimensions into 1-dimensional array: (2, 4, 1024)-->(2*4*1024)-->8192
model.add(Dense(2048, activation='relu', name='Class_2'))  # Dense layer changing neuron quantity 8192-->2048
model.add(Dense(4096, activation='relu', name='Class_3'))  # Dense layer changing neuron quantity 2048-->4096
model.add(Dense(CLASS_COUNT, activation='softmax', name='Class_4'))  # output dense layer classifying 4096 into 3 classes
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',  # used for output of more than 2 classes ('sparce_CE' takes in class labels in 'discrete numbers' view, not in one-hot-encoding)
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# let's start the learning:
'''store_learning = model.fit(x_data,
                           y_data,
                           validation_split=0.2,
                           shuffle=True,
                           batch_size=25,
                           epochs=35,
                           verbose=1)'''

print('first image is ', data_files[0])

i = 101
img = Image.open(data_files[i])

print('Default image size is: ', img.size)


# function to see the image
def show_image(img):
    plt.figure(figsize=(8, 5))  # creating the canvas
    plt.imshow(img)  # drawing the image
    plt.axis('off')  # not to see the unneeded axis
    plt.show()  # look what we've got


# function to compare 2 images: initial and augmented
def show_image_pair(img1, img2):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # creating the canvas to draw both images
    axs[0].imshow(img1)  # the initial image
    axs[0].axis('off')
    axs[1].imshow(img2)  # the augmented image
    axs[1].axis('off')
    plt.show()


show_image(img)

img_ = img.resize((256, 128))
print('The size of resized image is', img_.size)
show_image(img_)
# cropping the image: .crop takes 1 argument of tuple of 4: coordinates of left upper corner and right lower corner
img_crop = img_.crop((20, 10, 200, 100))
show_image(img_crop)
print(img_crop.size)


# random image crop function:
def random_crop(x,  # img
                f_x,  # limits of cropping in axis X
                f_y):  # limits of cropping in axis Y
    left = x.width * random.random() * f_x  # getting the right and left limits of cropping
    right = x.width * (1. - random.random() * f_x) - 1.
    upper = x.height * random.random() * f_y  # getting the upper and lower limits of cropping
    lower = x.height * (1. - random.random() * f_y) - 1.

    return x.crop((left, upper, right, lower))


img_crop = random_crop(img_, 0.2, 0.2)
show_image_pair(img_, img_crop)
print(img_crop.size)

angle = 10
img_rot = img_.rotate(angle, expand=True)
print(show_image(img_rot))


# function to calculate height and width of the biggest rectangle after
# rotating the initial rectangle (angle is the 3-d argument):
def rotated_rect(w, h, angle):
    angle = math.radians(angle)
    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


crop_w, crop_h = rotated_rect(img_width, img_height, angle)



