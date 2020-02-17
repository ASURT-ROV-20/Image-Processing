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


def edge_detect(frame):
    frame = cv2.Canny(frame, 100, 100)
    blur5 = cv2.GaussianBlur(frame,(5,5),0)
    blur3 = cv2.GaussianBlur(frame,(1,1),0)
    return blur5 - blur3


def color_compare(color1, color2):
    r = abs(int(color1[0]) - int(color2[0]))
    g = abs(int(color1[1]) - int(color2[1]))
    b = abs(int(color1[2]) - int(color2[2]))
    total = r+g+b
    if r < 100 and g < 100 and b < 100:
        print(color1)
        print(color2)
        print(total)
        print()
        return True
    else:
        print(color2)
        print("fail")
        return False


def get_colors(img):
    h, w, _ = img.shape
    roiTop = img[0:int(h/20), int(w/2) - int(w/20):int(w/2) + int(w/20)]
    roiBottom = img[18*int(h/20):h, int(w/2) - int(w/20):int(w/2) + int(w/20)]
    roiLeft = img[int(h/2) - int(h/20):int(h/2) + int(h/20), 0:int(w/20)]
    roiRight = img[int(h/2) - int(h/20):int(h/2) + int(h/20), 19*int(w/20):w]
    b, g, r, _ = np.uint8(cv2.mean(roiTop))
    top = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiBottom))
    bottom = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiRight))
    right = [r, g, b]
    b, g, r, _ = np.uint8(cv2.mean(roiLeft))
    left = [r, g, b]
    return [top, bottom, left, right]


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
    box = cnts_t[1]
    x, y, w, h = cv2.boundingRect(box)
    imgs[i] = imgs[i][y:y+h, x:x+w]
    colors.append(get_colors(imgs[i]))


# check bottom color to be not white (top side) and puts it in the beggining of the array
for i in range(0, 5):
    if not color_compare(colors[i][1], [255, 255, 255]):
        swap(0, i)
        break

# gets second plate under the top and puts it in the second place in the array
for i in range(1, 5):
    current = colors[0][1]  # top side bottom color
    if color_compare(current, colors[i][0]):  # compare to top colors in all sides
        swap(1, i)
        break

# gets plate on the side of the second plate and puts it in third place
for i in range(2, 5):
    current = colors[1][3]  # second side right color
    if color_compare(current, colors[i][2]):  # compare to left colors in all sides
        swap(2, i)
        break

for i in range(3, 5):
    current = colors[2][3]  # third side right color
    if color_compare(current, colors[i][2]):  # compare to left colors in all sides
        swap(3, i)
        break


# get final image dimentions
widthf = 0
heightf = 0
heightf, _, _ = imgs[0].shape
h, _, _ = imgs[1].shape
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
cv2.waitKey(0)
