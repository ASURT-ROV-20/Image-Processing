import numpy as np
import cv2
import math


def process_colar(image):
    height, width, _ = image.shape
    abstracted_coral = np.zeros((height,width,3), np.uint8)              # Empty image to draw on

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

    ############ Getting contour arround the whole coral and calculate the erosion Constant ##############
    
    cott, _ = cv2.findContours(mask,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

    contours_Dic = {}
    for i in range(len(cott)):
        area = cv2.contourArea(cott[i])
        contours_Dic[area] = i

    i = contours_Dic[max(list(contours_Dic))]
    
    _, _, coral_width, _ = cv2.boundingRect(cott[i])
    ######################################################################################################

    #######################   Applying Morphological Transformations   ###################################

    # 1- Closing Operation to remove small holes in the image (black holes in the mask) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # edges = cv2.Canny(pink_mask,100, 255)

    # 2- Erosion to flaten the surface of the coral 
    #    the surface isn't smooth in the linage between two pipes 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(coral_width//15,1))
    horizontal = cv2.erode(mask,kernel)               # horizontal Erosion to flaten Vertical surface 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,coral_width//14))
    vertical = cv2.erode(mask,kernel)                 # Vertical Erosion to flaten horizontal surface 
    ######################################################################################################


    ############################### Drawing Abstracted Coral #############################################
    hor_contours, _ = cv2.findContours(horizontal,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    hor_rectangles=[]
    for i in hor_contours:
        if cv2.contourArea(i) > 50:
            hor_rectangles.append(cv2.boundingRect(i))
            temp.append(i)
    hor_contours = temp
    for i in hor_rectangles:
        cv2.line(abstracted_coral, (i[0],int(i[1]+i[3]/2)),(i[0]+i[2],int(i[1]+i[3]/2)), (255,255,255), 5)


    ver_contours, _ = cv2.findContours(vertical,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    ver_rectangles=[]
    for i in ver_contours:
        if cv2.contourArea(i) > 20:
            ver_rectangles.append(cv2.boundingRect(i))
            temp.append(i)
    ver_contours = temp


    for i in ver_rectangles:
        cv2.line(abstracted_coral, (i[0]+int(i[2]/2),i[1]),(i[0]+int(i[2]/2),i[1]+i[3]), (255,255,255), 5)

    ### test ###
    # cv2.imshow("hor",image)
    # cv2.imshow("ver",vertical)
    # cv2.imshow("temp",original)
    # cv2.imshow("mask",mask)
    # cv2.imshow("abstract",abstracted_coral)
    # cv2.waitKey(0)
        
    ######################################################################################################

    ################################# Getting the points #################################################
    max_y=0
    for i in hor_rectangles:
        if( int(i[1]+i[3]/2) > max_y):
            max_y = int(i[1]+i[3]/2)
            x_of_max_y = i[0]
    #cv2.circle(abstracted_coral,(x_of_max_y,max_y),3,(0,0,255),2) # test
    desired_y = max_y

    max_y=0
    x1_of_max_y=0
    x2_of_max_y=0
    reversed_ver_rectangles = list(reversed(ver_rectangles))

    for i in ver_rectangles:
        if( (i[1]+i[3]) > max_y):
            max_y = i[1]+i[3]
            x1_of_max_y = i[0]+int(i[2]/2)
    #cv2.circle(abstracted_coral,(x1_of_max_y,max_y),3,(0,0,255),2) # test

    max_y=0
    for i in reversed_ver_rectangles:
        if( (i[1]+i[3]) > max_y):
            max_y = i[1]+i[3]
            x2_of_max_y = i[0]+int(i[2]/2)
    #cv2.circle(abstracted_coral,(x2_of_max_y,max_y),3,(0,0,255),2) #test
    return abstracted_coral, x1_of_max_y, x2_of_max_y, desired_y



    ######################################################################################################

def main():
    image = cv2.imread("destination.PNG")
    abstracted_coral1, abstracted_coral1_x1, abstracted_coral1_x2, abstracted_coral1_y = process_colar(image)
    abstracted_coral1_distance = abs(abstracted_coral1_x1 - abstracted_coral1_x2)
    y, x, _ = image.shape
    # cv2.circle(abstracted_coral1,(abstracted_coral1_x1,abstracted_coral1_y),3,(0,0,255),2)
    # cv2.circle(abstracted_coral1,(abstracted_coral1_x2,abstracted_coral1_y),3,(0,0,255),2)

    image = cv2.imread("1.PNG")
    abstracted_coral2, abstracted_coral2_x1, abstracted_coral2_x2, abstracted_coral2_y = process_colar(image)
    abstracted_coral2_distance = abs(abstracted_coral2_x1 - abstracted_coral2_x2)
    print(abstracted_coral1_distance,abstracted_coral2_distance)
    # cv2.circle(abstracted_coral2,(abstracted_coral2_x1,abstracted_coral2_y),3,(0,0,255),2)
    # cv2.circle(abstracted_coral2,(abstracted_coral2_x2,abstracted_coral2_y),3,(0,0,255),2)

    scale_factor = abstracted_coral1_distance/abstracted_coral2_distance
    height, width, _ = abstracted_coral2.shape
    scaled_abstracted_coral2 = cv2.resize(abstracted_coral2,(int(scale_factor*width),int(scale_factor*height)))
    _,scaled_abstracted_coral2 = cv2.threshold(scaled_abstracted_coral2,10,255,cv2.THRESH_BINARY)

    tx = abstracted_coral1_x1 - scale_factor * abstracted_coral2_x1
    ty = abstracted_coral1_y - scale_factor * abstracted_coral2_y
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    img_translation = cv2.warpAffine(scaled_abstracted_coral2, T, (x, y))
    
    # result = abstracted_coral1 - img_translation

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    result = cv2.erode(abstracted_coral1 - img_translation,kernel)

    # cv2.imshow("destination",abstracted_coral1)
    # cv2.imshow("transformed",abstracted_coral2)
    cv2.imshow("translated", result)
    # cv2.imshow("result",result)
    cv2.waitKey(0)


if __name__== "__main__":
  main()


############## Showing Images ###########
# cv2.imshow("hor",image)
# cv2.imshow("ver",vertical)
# cv2.imshow("temp",original)
# cv2.imshow("mask",mask)
# cv2.imshow("abstract",abstracted_coral)
# cv2.waitKey(0)

############ For Testing #################
# boundRect = ver_rectangles
# for i in range(len(boundRect)):
#     cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])),
#           (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,255), 2)

# cv2.drawContours(image, hor_contours, -1, (0,255,0), 9)

# TODO hit or miss to implement adaptive erosion