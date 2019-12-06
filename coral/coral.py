import numpy as np
import cv2
import math

image = cv2.imread("Capture.PNG")

abstracted_coral = np.zeros((424,387,3), np.uint8)              # Empty image to draw on

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

################################ Another approch #####################################################

x, y, _ = image.shape
print(math.ceil(x / 25))
print(math.ceil(y / 25))

######################################################################################################
'''
#######################   Applying Morphological Transformations   ###################################

# 1- Closing Operation to remove small holes in the image (black holes in the mask) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# edges = cv2.Canny(pink_mask,100, 255)

# 2- Erosion to flaten the surface of the coral 
#    the surface isn't smooth in the linage between two pipes 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,1))
horizontal = cv2.erode(mask,kernel)               # horizontal Erosion to flaten Vertical surface 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,25))
vertical = cv2.erode(mask,kernel)                 # Vertical Erosion to flaten horizontal surface 
######################################################################################################

hor_contours, _ = cv2.findContours(horizontal,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
temp=[]
hor_rectangles=[]
for i in hor_contours:
    if cv2.contourArea(i) > 100:
        hor_rectangles.append(cv2.boundingRect(i))
        temp.append(i)
hor_contours = temp
# cv2.drawContours(image, hor_contours, -1, (0,255,0), 9)
for i in hor_rectangles:
    cv2.rectangle(abstracted_coral, (i[0],int(i[1]+i[3]/2)),(i[0]+i[2],int(i[1]+i[3]/2)), (255,255,255), 2)


ver_contours, _ = cv2.findContours(vertical,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
temp=[]
ver_rectangles=[]
for i in ver_contours:
    if cv2.contourArea(i) > 100:
        ver_rectangles.append(cv2.boundingRect(i))
        temp.append(i)
ver_contours = temp
for i in ver_rectangles:
    cv2.rectangle(abstracted_coral, (i[0]+int(i[2]/2),i[1]),(i[0]+int(i[2]/2),i[1]+i[3]), (255,255,255), 2)

# boundRect = ver_rectangles
# for i in range(len(boundRect)):
#     cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])),
#           (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,255), 2)

cv2.imshow("hor",vertical)
cv2.imshow("ver",abstracted_coral)
cv2.imshow("pink",image)
cv2.waitKey(0)

# lines = cv2.HoughLinesP(edges,1,np.pi/180,40,maxLineGap=15)
# print(len(lines)/2)
# if lines is not None :
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
'''