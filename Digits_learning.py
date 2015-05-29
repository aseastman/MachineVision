#This code uses OpenCV to predict digits using kNN

import numpy as np
import cv2
from matplotlib import pyplot as plt

#Read in the image as grayscale 
img = cv2.cvtColor(cv2.imread('digits.png'),cv2.COLOR_BGRA2GRAY)

#Separate each number in the image
cells = np.array([np.hsplit(row,100) for row in np.vsplit(img,50)])

#Split the numbers into training and testing data
train = cells[:,:25].reshape(-1,400).astype(np.float32)
test = cells[:,25:100].reshape(-1,400).astype(np.float32)

#Create the labels for each number in the training and testing data
train_labels = np.repeat(np.arange(10),125)[:,np.newaxis]
test_labels = np.repeat(np.arange(10),375)[:,np.newaxis]

for k in range(1,8):
#Do the kNN solver and output the accuracy of the test results.
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    ret,result,neighbours,dist = knn.find_nearest(test,k)
    
    print "The accuracy of the kNN solver for the digits image is %.2f."\
    % (np.count_nonzero(result==test_labels)*100.0/result.size)
    
    plt.plot(k,(np.count_nonzero(result==test_labels)*100.0/result.size),'o')
    plt.xlabel('k-value')
    plt.ylabel('Accuracy (%)')
    plt.xlim((0.9,7.1))