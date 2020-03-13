import cv2
import math
import numpy as np

def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
    return len(approx)

def get_blue_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 130, 50])
    upper_blue = np.array([130, 255, 200])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    return thresh
    
def get_largest_two_contours(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if get_vertex_count(c) == 2]
    largest_two_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    return largest_two_contours

def get_blue_line_contours(frame):
    # frame = cv2.resize(frame, (600, 400))
    thresh = get_blue_mask(frame)
    return get_largest_two_contours(thresh)

def midpoint(cnt):
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    mid = (bottommost[0] + topmost[0]) / 2, (bottommost[1] + topmost[1]) / 2
    return mid

def get_distance(cnt1, cnt2):
    m1 = midpoint(cnt1)
    m2 = midpoint(cnt2)
    distance = math.sqrt(pow(m1[0] - m2[0], 2) + pow(m1[1] - m2[1], 2))
    return distance
