import math

import numpy as np
import cv2

img = cv2.imread('star.png')
img = cv2.GaussianBlur(img, (3, 3), 0)
img2 = cv2.imread('star2.png')
img2 = cv2.GaussianBlur(img2, (3, 3), 0)

t1 = 100
t2 = 200
min_cnt = 8

def findAndDrawContours(name, src):
    global t1, t2
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, t1, t2)
    cv2.namedWindow(name + " Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name + " Edges", 500, 500)
    cv2.imshow(name + " Edges", canny)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_final = src.copy()
    for cnt in contours:
        if (cv2.contourArea(cnt) > min_cnt):
            cv2.drawContours(img_final, [cnt], -1, (0, 255, 0), 3)

    cv2.namedWindow(name + " Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name + " Contours", 500, 500)
    cv2.imshow(name + " Contours", img_final)

def displayContours():
    findAndDrawContours("Star1", img)
    findAndDrawContours("Star2", img2)

def onT1Changed(v):
    global t1
    t1 = v
    displayContours()

def onT2Changed(v):
    global t2
    t2 = v
    displayContours()

def onMinCntChanged(v):
    global min_cnt
    min_cnt = v
    displayContours()


cv2.namedWindow("T")
cv2.createTrackbar("T1", "T", t1, 255, onT1Changed)
cv2.createTrackbar("T2", "T", t2, 255, onT2Changed)
cv2.createTrackbar("Min Contour", "T", min_cnt, 200, onMinCntChanged)
displayContours()

cv2.waitKey()
cv2.destroyAllWindows()
