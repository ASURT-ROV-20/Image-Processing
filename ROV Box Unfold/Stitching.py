import cv2
import numpy as np


def swap(i, j):
    global colors
    global imgs
    temp = imgs[i].copy()
    imgs[i] = imgs[j].copy()
    imgs[j] = temp.copy()
    temp = colors[i]
    colors[i] = colors[j]
    colors[j] = temp


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
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
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def edge_detect(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # take a sample of the white color in the middle of the image
    ht, wt, _ = hsv.shape
    roi = hsv[int(ht / 2) - int(ht / 30):int(ht / 2) + int(ht / 30), int(wt / 2) - int(wt / 30):int(wt / 2) + int(wt / 30)]
    h, s, v, _ = np.uint8(cv2.mean(roi))
    # mask the white color
    lower = np.array([h-20, s-20, v-40])
    upper = np.array([h+20, s+20, v+40])
    mask = cv2.inRange(hsv, lower, upper)
    frame = cv2.Canny(mask, 100, 100)
    blur5 = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
    blur3 = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    result = blur5-blur3
    return result


def color_compare(color1, color2):
    r = abs(int(color1[0]) - int(color2[0]))
    g = abs(int(color1[1]) - int(color2[1]))
    b = abs(int(color1[2]) - int(color2[2]))
    if r < 50 and g < 50 and b < 50:
        return True
    else:
        return False


def get_colors(imgc):
    hc, wc, _ = imgc.shape

    roiTop = imgc[0:int(hc/20), int(wc/2) - int(wc/20):int(wc/2) + int(wc/20)]
    roiBottom = imgc[18*int(hc/20):hc, int(wc/2) - int(wc/20):int(wc/2) + int(wc/20)]
    roiLeft = imgc[int(hc/2) - int(hc/20):int(hc/2) + int(hc/20), 0:int(wc/20)]
    roiRight = imgc[int(hc/2) - int(hc/20):int(hc/2) + int(hc/20), 19*int(wc/20):wc]
    roiCenter = imgc[int(hc / 2) - int(hc / 30):int(hc / 2) + int(hc / 30), int(wc / 2) - int(wc / 30):int(wc / 2) + int(wc / 30)]

    b, g, r, _ = np.uint8(cv2.mean(roiTop))
    top = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiBottom))
    bottom = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiRight))
    right = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiLeft))
    left = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiCenter))
    center = [r, g, b]
    return [top, bottom, left, right, center]


imgs = []
colors = []

# load images from file
for i in range(1, 6):
    img = cv2.imread("./Images/Box" + str(i) + ".jpg")
    imgs.append(img)

# extract boxes from images
for i in range(0, 5):
    temp = imgs[i]
    imgs[i] = cv2.GaussianBlur(temp, (3, 3), cv2.BORDER_DEFAULT)
    # edges
    edges = edge_detect(imgs[i])
    # find the contours
    cnts_t, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # SORT
    cnts_t = sorted(cnts_t, key=cv2.contourArea, reverse=True)
    # Get box contour
    box_cnt = cnts_t[0]
    epsilon = 0.1 * cv2.arcLength(box_cnt, True)
    box_cnt = cv2.approxPolyDP(box_cnt, epsilon, True)

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

    colors.append(get_colors(imgs[i]))


# check bottom color to be not white (top side) and puts it in the beginning of the array
for i in range(0, 5):
    print("Checking box " + str(i) + " for non white bottom color (top box)")
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

for i in range(3, 5):
    print("Comparing box 2 right, color (" + str(colors[2][3]) + ") with box " + str(i) + " left color (" + str(colors[i][2]) + ")")
    current = colors[2][3]  # third side right color
    if color_compare(current, colors[i][2]):  # compare to left colors in all sides
        print("found it")
        swap(3, i)
        break

# Correct image heights to match
for i in range(0, 5):
    imgs[i] = image_resize(imgs[i], height=200)
imgs[0] = image_resize(imgs[0], height=imgs[1].shape[0])

# Write image numbers on them
for i in range(0, 5):
    w, h, _ = imgs[i].shape
    cv2.putText(imgs[i], str(i), ((h/2), (w/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 50), 1)

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
final[0:h1, imgs[1].shape[1]:imgs[1].shape[1]+w1] = imgs[0]

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
cv2.waitKey(0)
