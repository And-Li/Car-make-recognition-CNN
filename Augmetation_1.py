from PIL import Image
import matplotlib.pyplot as plt
from Car_recognition_1 import data_files

i = 100
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
    fig, axs = plt.subplots(1, 2, igsize=(14, 5))  # creating the canvas to draw both images
    axs[0].imshow(img1)  # the initial image
    axs[0].axis('off')
    axs[1].imshow(img2)  # the augmented image
    axs[1].axos('off')
    plt.show()

