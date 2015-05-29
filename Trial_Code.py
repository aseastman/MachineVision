#This simple code reads in a picture in grayscale
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test_pic.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()