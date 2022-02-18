import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('C:/NN/NIST SD 19/DataSet/77-93.png', 0)
# img = cv.imread('C:\Users\manas\OneDrive\STAT 350\Homework 2 Problem 4.png', 0)
edges = cv.Canny(img,100,200)

plt.subplot(121) 
plt.imshow(img,cmap = 'gray')
plt.title('Original Image') 
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
