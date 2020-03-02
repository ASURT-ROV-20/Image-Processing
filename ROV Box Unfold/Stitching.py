import cv2
import numpy as np
import time

def swap(i, j):
    # Swaps two boxes images and their corresponding colors in their arrays
    global colors
    global imgs
    temp = imgs[i].copy()
    imgs[i] = imgs[j].copy()
    imgs[j] = temp.copy()
    temp = colors[i]
    colors[i] = colors[j]
    colors[j] = temp


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def edge_detect(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    # take a sample of the white color in the middle of the image, convert image to HSV for better ranges
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ht, wt, _ = hsv.shape
    roi = hsv[int(ht / 2) - int(ht / 30):int(ht / 2) + int(ht / 30), int(wt / 2) - int(wt / 30):int(wt / 2) + int(wt / 30)]
    h, s, v, _ = np.uint8(cv2.mean(roi))
    # mask the white color
    lower = np.array([h-40, s-40, v-40])
    upper = np.array([h+40, s+40, v+40])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    outmask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=15)
    colormask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    outmask = cv2.erode(outmask, kernel, iterations=1)
    colormask = cv2.dilate(colormask, kernel, iterations=2)

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        red = np.zeros(temp.shape, temp.dtype)
        red[:, :] = (0, 0, 255)
        maskoverlay = np.bitwise_and(red, outmask[:, :, np.newaxis])
        cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
        cv2.putText(temp, "Box Mask", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)
        temp = frame.copy()
        blue = np.zeros(temp.shape, temp.dtype)
        blue[:, :] = (255, 0, 0)
        maskoverlay = np.bitwise_and(blue, colormask[:, :, np.newaxis])
        cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
        cv2.putText(temp, "Colors Mask", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)
    # DEBUG !!!!

    # Get the mask edges
    outeredge = cv2.Canny(outmask, 5, 5)
    coloredges = cv2.Canny(colormask, 5, 5)

    # DEBUG !!!!
    if __debug__:
        temp = np.bitwise_or(frame, outeredge[:, :, np.newaxis])
        temp = np.bitwise_or(temp, coloredges[:, :, np.newaxis])
        maskoverlay = np.bitwise_and(red, outeredge[:, :, np.newaxis])
        maskoverlay2 = np.bitwise_and(blue, coloredges[:, :, np.newaxis])
        cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
        cv2.addWeighted(temp, 0.5, maskoverlay2, 1, 1, temp)
        cv2.putText(temp, "Edges (Blue for colors Red for box)", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)
    # DEBUG !!!!

    return outeredge, coloredges


def color_compare(color1, color2):
    h = abs(int(color1[0]) - int(color2[0]))
    s = abs(int(color1[1]) - int(color2[1]))
    v = abs(int(color1[2]) - int(color2[2]))
    if h < 20 and v < 25 and s < 25:
        return True
    elif h > 150 and (((color1[0] < 15) and (color2[0] > 160)) or ((color2[0] < 15) and (color1[0] > 160))) and v < 30:
        return True
    else:
        return False


def get_colors(imgc, cnts):
    imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2HSV)
    boxX, boxY, wc, hc = cv2.boundingRect(cnts[0])
    hasBottom = False
    cnts = cnts[2:]
    for c in cnts:
        area = cv2.contourArea(c,True)
        if area > 0:
            x,y,w,h = cv2.boundingRect(c)
            print([x,y,w,h])
            print([boxX, boxY, wc,hc])
        else:
            continue
        roiCenter = imgc[boxY + int(hc / 2) - int(hc / 30):boxY + int(hc / 2) + int(hc / 30), boxX + int(wc / 2) - int(wc / 30):boxX + int(wc / 2) + int(wc / 30)]
        hh, s, v, _ = np.uint8(cv2.mean(roiCenter))
        center = [hh, s, v]
        mask = np.zeros(imgc.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, 1)
        cv2.rectangle(imgc,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("test", imgc)
        cv2.waitKey(0)
        cv2.imshow("test", mask)
        cv2.waitKey(0)
        if (x-boxX) < wc/10:
            h, s, v, _ = np.uint8(cv2.mean(imgc[y:y+h,x:x+w]))
            print([h,s,v])
            left = [h, s, v]
        elif (x-boxX) > wc/2:
            h, s, v, _ = np.uint8(cv2.mean(imgc[y:y+h,x:x+w]))
            print([h,s,v])
            right = [h, s, v]
        elif (y-boxY) < hc/10:
            h, s, v, _ = np.uint8(cv2.mean(imgc[y:y+h,x:x+w]))
            print([h, s, v])
            top = [h, s, v]
        elif (y-boxY) > hc/2:
            h, s, v, _ = np.uint8(cv2.mean(imgc[y:y+h,x:x+w]))
            print([h, s, v])
            hasBottom = True
            bottom = [h, s, v]
    if not hasBottom:
        bottom = center.copy()
    return [top, bottom, left, right, center]


imgs = []
colors = []

startTime = time.time()

# load images from file
for i in range(1, 6):
    img = cv2.imread("./Images/Box" + str(i) + ".jpg")
    imgs.append(img)

# extract boxes from images
for i in range(0, 5):
    temp = imgs[i]
    imgs[i] = cv2.GaussianBlur(temp, (3, 3), cv2.BORDER_DEFAULT)
    # edges
    outerEdge, colorsEdges = edge_detect(imgs[i])
    # find the box contour
    cnts_b, _ = cv2.findContours(outerEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # SORT
    cnts_b = sorted(cnts_b, key=cv2.contourArea, reverse=True)
    # Get box contour
    box_cnt = cnts_b[0]
    # find colors contours
    cnts_c, _ = cv2.findContours(colorsEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # remove box contour
    cnts_c = sorted(cnts_c, key=cv2.contourArea, reverse=True)

    angle = cv2.minAreaRect(box_cnt)[-1]

    print(angle)

    (h, w) = imgs[i].shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    imgs[i] = cv2.warpAffine(imgs[i], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    colors.append(get_colors(imgs[i], cnts_c))

    cnts_c = cnts_c[2:]

    # DEBUG !!!!
    if __debug__:
        temp = imgs[i].copy()
        cv2.drawContours(temp, [box_cnt], -1, (0, 0, 255), 1)
        cv2.drawContours(temp, cnts_c, -1, (255, 0, 0), 1)
        cv2.putText(temp, "Contours (Blue for colors Red for box)", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)
    # DEBUG !!!!

    # Approximate contour to rectangle (hopefully)
    epsilon = 0.07 * cv2.arcLength(box_cnt, True)
    box_cnt = cv2.approxPolyDP(box_cnt, epsilon, True)

    # DEBUG !!!!
    if __debug__:
        temp = imgs[i].copy()
        cv2.drawContours(temp, box_cnt, -1, (0, 255, 0), 5)
        cv2.putText(temp, "Box Approximation", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)
    # DEBUG !!!!

    # get edge points for perspective warp correction
    pts = box_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Make a new image with correct size to store the warped image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    # Get and apply perspective warp correction
    M = cv2.getPerspectiveTransform(rect, dst)
    imgs[i] = cv2.warpPerspective(imgs[i], M, (maxWidth, maxHeight))

    # Get and append box colors to their array

print(colors)
# check bottom color to be not white (top side) and puts it in the beginning of the array
for i in range(0, 5):
    print("Checking box " + str(i) + " for non white bottom color (top box)")
    print([colors[i][1], colors[i][4]])
    if not color_compare(colors[i][1], colors[i][4]):
        print("found it")
        swap(0, i)
        break

# gets second plate under the top and puts it in the second place in the array
for i in range(1, 5):
    print("Comparing box 0 bottom, color (" + str(colors[0][1]) + ") with box " + str(i) + " top color ("+ str(colors[i][0]) + ")")
    current = colors[0][1]  # top side bottom color
    if color_compare(current, colors[i][0]):  # compare to top colors in all sides
        print("found it")
        swap(1, i)
        break

# gets plate on the side of the second plate and puts it in third place
for i in range(2, 5):
    print("Comparing box 1 right, color (" + str(colors[1][3]) + ") with box " + str(i) + " left color (" + str(colors[i][2]) + ")")
    current = colors[1][3]  # second side right color
    if color_compare(current, colors[i][2]):  # compare to left colors in all sides
        print("found it")
        swap(2, i)
        break

# gets plate on the side of the third plate and puts it in fourth place
for i in range(3, 5):
    print("Comparing box 2 right, color (" + str(colors[2][3]) + ") with box " + str(i) + " left color (" + str(colors[i][2]) + ")")
    current = colors[2][3]  # third side right color
    if color_compare(current, colors[i][2]):  # compare to left colors in all sides
        print("found it")
        swap(3, i)
        break

# The fifth plate is in its correct position as t is the only remaining one

# Correct image heights to match
for i in range(0, 5):
    imgs[i] = image_resize(imgs[i], height=200)
imgs[0] = image_resize(imgs[0], width=imgs[1].shape[1])

# Write image numbers on them
for i in range(0, 5):
    w, h, _ = imgs[i].shape
    cv2.putText(imgs[i], str(i), (int(h/2), int(w/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 50), 1)

# get final image dimensions
widthf = 0
heightf, _, _ = imgs[0].shape
h, _, _ = imgs[2].shape
heightf = heightf + h
for i in range(1, 5):
    _, w, _ = imgs[i].shape
    widthf = w + widthf

# make final image
final = np.zeros((heightf, widthf, 3), np.uint8)
final = 255 - final


# layout images on final image
delx = 0
h1, w1, _ = imgs[0].shape
final[0:h1, 0:w1] = imgs[0]

h2, w2, _ = imgs[1].shape
final[heightf-h2:heightf, 0:delx+w2] = imgs[1]
delx = delx + w2

h3, w3, _ = imgs[2].shape
final[heightf-h3:heightf, delx:delx+w3] = imgs[2]
delx = delx + w3

h4, w4, _ = imgs[3].shape
final[heightf-h4:heightf, delx:delx+w4] = imgs[3]
delx = delx + w4

h5, w5, _ = imgs[4].shape
final[heightf-h5:heightf, delx:delx+w5] = imgs[4]

cv2.imshow("Final", final)
print(" ---- %s seconds ----" % (time.time()-startTime))
cv2.waitKey(0)
