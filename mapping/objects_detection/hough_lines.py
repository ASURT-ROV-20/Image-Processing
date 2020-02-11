import math

import numpy as np
import cv2

img = cv2.imread('star.png')
img = cv2.GaussianBlur(img, (3, 3), 0)
img2 = cv2.imread('star2.png')
img2 = cv2.GaussianBlur(img2, (3, 3), 0)

t1 = 100
t2 = 200

def findAndDrawEdges(name, src):
    global t1, t2
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, t1, t2)
    cv2.namedWindow(name + " Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name + " Edges", 500, 500)
    cv2.imshow(name + " Edges", canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 80, None, 0, 0)
    img_final = src.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img_final, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.namedWindow(name + " Lines", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name + " Lines", 500, 500)
    cv2.imshow(name + " Lines", img_final)

def displayEdges():
    findAndDrawEdges("Star1", img)
    findAndDrawEdges("Star2", img2)

def onT1Changed(v):
    global t1
    t1 = v
    displayEdges()

def onT2Changed(v):
    global t2
    t2 = v
    displayEdges()


cv2.namedWindow("T")
cv2.createTrackbar("T1", "T", 0, 255, onT1Changed)
cv2.createTrackbar("T2", "T", 0, 255, onT2Changed)
displayEdges()

cv2.waitKey()
cv2.destroyAllWindows()
