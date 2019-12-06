import numpy as np
import cv2
import math

def draw(array, cellSize):
    x, y = array.shape
    abstracted = np.zeros((y*cellSize,x*cellSize), np.uint8)
    for row in range(x):
        for col in range(y):
            color = array[row][col]
            # print(type(int(4)))
            cv2.rectangle(abstracted, (col*cellSize, row*cellSize), ((col+1)*cellSize, (row+1)*cellSize),int(color), cv2.FILLED)
            # print(array[row][col])
    return abstracted

image = cv2.imread("Capture.PNG")

abstracted_coral = np.zeros((387,424,3), np.uint8)              # Empty image to draw on

#########################################    Color Mask    ###########################################
hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)                      
lower_pink= np.array([150,80,50])
higher_pink = np.array([170,255,255])
pink_mask= cv2.inRange(hsv,lower_pink,higher_pink)

lower_white= np.array([0,0,210])
higher_white = np.array([255,255,255])
white_mask= cv2.inRange(hsv,lower_white,higher_white)
                                                                                                    
mask = pink_mask + white_mask                                                                       
######################################################################################################

#######################   Applying Morphological Transformations   ###################################

# 1- Closing Operation to remove small holes in the image (black holes in the mask) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,1))
horizontal = cv2.erode(mask,kernel)               # horizontal Erosion to flaten Vertical surface 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,25))
vertical = cv2.erode(mask,kernel)                 # Vertical Erosion to flaten horizontal surface 

mask = vertical + horizontal

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


######################################################################################################

################################ Another approch #####################################################

x, y, _ = image.shape                               # shape of the orignal image
gWidth = 20                                         # width of the grid cell
VLineCount = math.floor(x / gWidth)                 # number of vertical lines (cols)
HLineCount = math.floor(y / gWidth)                 # number of horizontal lines (rows)                           
output = np.zeros(((HLineCount+1),                  # output image
                    (VLineCount+1)), np.uint8)


for row in range(HLineCount+1):
    for col in range(VLineCount+1):
        part = mask[row*gWidth:(row+1)*gWidth, col*gWidth:(col+1)*gWidth]
        print(round((np.sum(part)/255)/(gWidth*gWidth),1)," ", end="")
        output[row][col] = (np.sum(part)/255)/(gWidth*gWidth)*255
    print("\n")
kernel = np.array([[0,1,0],
                   [1,10,1],
                   [0,1,0]])
kernel = kernel / 14
out2 = cv2.filter2D(output, -1, kernel)
resized = cv2.resize(out2, (500,500),interpolation = cv2.INTER_NEAREST)
out2 = draw(out2,10)
cv2.imshow("out2",out2)
_,out2 = cv2.threshold(out2,255*0.4,255,cv2.THRESH_BINARY)
cv2.imshow("out88",out2)

# Draw  the grid on the image
for vline in range(HLineCount):
    cv2.line(mask,((vline+1)*gWidth,0),((vline+1)*gWidth,x),(255,255,255),1)
for hline in range(VLineCount):
    cv2.line(mask,(0,(hline+1)*gWidth),(y,(hline+1)*gWidth),(255,255,255),1)

cv2.imshow("mask",mask)
cv2.waitKey(0)

######################################################################################################
