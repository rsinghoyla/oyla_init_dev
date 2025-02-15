import cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)
num_frames = 100
import time
then = time.time()
movie = []
for _ in range(num_frames):
    bleh, img = cap.read()

    #laplacian = cv2.Laplacian(img,cv2.CV_64F)
    #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    #plt.subplot(2,2,1),
    plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.pause(0.0001)
    plt.clf()
now = time.time()
print("Frame rate: ", (num_frames/(now-then)))
plt.show()
