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


def crop_Box(frame):
    # resize the image and blur it for better recognition
    frame = image_resize(frame, width=700)
    frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)

    # take a sample of the white color in the middle of the image, convert image to HSV for better ranges
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ht, wt, _ = hsv.shape
    roi = hsv[int(ht / 2) - int(ht / 30):int(ht / 2) + int(ht / 30), int(wt / 2) - int(wt / 30):int(wt / 2) + int(wt / 30)]
    hue, sat, val, _ = np.uint8(cv2.mean(roi))

    # mask the white color
    lower = np.array([hue-40, sat-40, val-40])
    upper = np.array([hue+40, sat+40, val+40])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)

    # clean the mask and close the color boxes
    outmask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=15)
    outmask = cv2.erode(outmask, kernel, iterations=1)

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

    # Get the mask edges
    outeredge = cv2.Canny(outmask, 5, 5)

    # DEBUG !!!!
    if __debug__:
        temp = np.bitwise_or(frame, outeredge[:, :, np.newaxis])
        maskoverlay = np.bitwise_and(red, outeredge[:, :, np.newaxis])
        cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
        cv2.putText(temp, "Box Edges", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    # get all contours
    cnts_b, _ = cv2.findContours(outeredge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # extract box contour
    cnts_b = sorted(cnts_b, key=cv2.contourArea, reverse=True)
    box_cnt = cnts_b[0]

    # correct angle
    angle = cv2.minAreaRect(box_cnt)[-1]
    if __debug__: print("Angle Correction: %s" % angle)
    if abs(angle)<30:
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        cv2.drawContours(temp, [box_cnt], -1, (0, 0, 255), 1)
        cv2.putText(temp, "Box Contour", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    # Approximate contour to box
    epsilon = 0.07 * cv2.arcLength(box_cnt, True)
    box_cnt = cv2.approxPolyDP(box_cnt, epsilon, True)

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        cv2.drawContours(temp, box_cnt, -1, (0, 255, 0), 5)
        cv2.putText(temp, "Box Approximation", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

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
    frame = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        cv2.putText(temp, "Cropped Box", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    # add white space for fail safe contour detection
    h, w, _ = frame.shape
    temp = np.zeros((h+10, w+10, 3), np.uint8)
    temp[:, :] = (hue, sat, val)
    temp = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    temp[5:h+5, 5:w+5] = frame
    frame = temp

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        cv2.putText(temp, "Added white space", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    return frame


def get_colors(frame):

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

    # clean the mask
    colormask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    colormask = cv2.dilate(colormask, kernel, iterations=2)

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        blue = np.zeros(temp.shape, temp.dtype)
        blue[:, :] = (255, 0, 0)
        maskoverlay = np.bitwise_and(blue, colormask[:, :, np.newaxis])
        cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
        cv2.putText(temp, "Colors Mask", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    # Get the mask edges
    coloredges = cv2.Canny(colormask, 5, 5)

    # DEBUG !!!!
    if __debug__:
        temp = np.bitwise_or(temp, coloredges[:, :, np.newaxis])
        maskoverlay = np.bitwise_and(blue, coloredges[:, :, np.newaxis])
        cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
        cv2.putText(temp, "Colors Edges", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    # get all contours
    cnts_c, _ = cv2.findContours(coloredges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # DEBUG !!!!
    if __debug__:
        temp = frame.copy()
        cv2.drawContours(temp, cnts_c, -1, (255, 0, 0), 1)
        cv2.putText(temp, "Colors Contours", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
        cv2.imshow("Debug", temp)
        cv2.waitKey(0)

    # Set all sides to white
    top = bottom = left = right = center = [h, s, v]
    hc, wc, _ = frame.shape
    for c in cnts_c:
        # Contours are doubled from canny edges but with negative area, so we will ignore processing negative areas
        area = cv2.contourArea(c, True)
        if area > 0:
            x, y, w, h = cv2.boundingRect(c)
        else:
            continue

        # Mask colors according to contour to get an accurate average color
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # Compare bounding box points to frame points to get position
        if x < (wc / 9):
            h, s, v, _ = np.uint8(cv2.mean(hsv, mask=mask))
            left = [h, s, v]
        elif x > (wc / 2):
            h, s, v, _ = np.uint8(cv2.mean(hsv, mask=mask))
            right = [h, s, v]
        elif y < (hc / 8):
            h, s, v, _ = np.uint8(cv2.mean(hsv, mask=mask))
            top = [h, s, v]
        elif y > (hc / 2):
            h, s, v, _ = np.uint8(cv2.mean(hsv, mask=mask))
            bottom = [h, s, v]

        # DEBUG !!!!
        if __debug__:
            temp = frame.copy()
            color = np.zeros(temp.shape, temp.dtype)
            color[:, :] = (0, 255, 0)
            maskoverlay = np.bitwise_and(color, mask[:, :, np.newaxis])
            cv2.addWeighted(temp, 0.5, maskoverlay, 1, 1, temp)
            cv2.putText(temp, "Color mask", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, 1)
            print("Contour Position: "+str([x, y, w, h]))
            print("Frame dimensions:" + str(hc) + ", " + str(wc))
            print("Detected color:" + str(h) + ", " + str(s) + ", " + str(v))
            cv2.imshow("Debug", temp)
            cv2.waitKey(0)

    return [top, bottom, left, right, center]


def color_compare(color1, color2):
    h = abs(int(color1[0]) - int(color2[0]))
    s = abs(int(color1[1]) - int(color2[1]))
    v = abs(int(color1[2]) - int(color2[2]))
    if h < 20 and v < 50 and s < 50:
        return True
    elif h > 150 and (((color1[0] < 15) and (color2[0] > 160)) or ((color2[0] < 15) and (color1[0] > 160))) and v < 30:
        return True
    else:
        return False

imgs = []
colors = []

startTime = time.time()

# load images from file
for i in range(1, 6):
    img = cv2.imread("./Images/Box" + str(i) + ".jpg")
    imgs.append(img)

# Process images
for i in range(0, 5):
    imgs[i] = crop_Box(imgs[i])
    colors.append(get_colors(imgs[i]))


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

# Set top image side to second side width
imgs[0] = image_resize(imgs[0], width=imgs[1].shape[1])

# Write image numbers on  them
if __debug__:
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

# Resize final image to fit screen
final = image_resize(final, width=1300)

cv2.imshow("Final", final)
print(" ---- Finished in %s seconds ----" % (time.time()-startTime))
cv2.waitKey(0)
