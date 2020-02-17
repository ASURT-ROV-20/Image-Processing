import numpy as np
import cv2
import math

def getMask(image):
    hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)                      
    lower_pink= np.array([150,80,50])
    higher_pink = np.array([170,255,255])
    pink_mask= cv2.inRange(hsv,lower_pink,higher_pink)

    lower_white= np.array([0,0,210])
    higher_white = np.array([255,255,255])
    white_mask= cv2.inRange(hsv,lower_white,higher_white)
    # cv2.imshow("WHITE", white_mask)
    # cv2.waitKey(0)                                                                     
    return pink_mask, white_mask        

def getErosionValue(mask):
    cott, _ = cv2.findContours(mask,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

    contours_Dic = {}
    for i in range(len(cott)):
        area = cv2.contourArea(cott[i])
        contours_Dic[area] = i

    i = contours_Dic[max(list(contours_Dic))]
    
    _, _, coral_width, _ = cv2.boundingRect(cott[i])
    return coral_width

def drawAbstractedCoral(image, vertical, horizontal):
    height, width, _ = image.shape
    abstracted_coral = np.zeros((height,width,3), np.uint8)              # Empty image to draw on
    hor_contours, _ = cv2.findContours(horizontal,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    hor_rectangles=[]
    for i in hor_contours:
        if cv2.contourArea(i) > 50:
            hor_rectangles.append(cv2.boundingRect(i))
            temp.append(i)
    hor_contours = temp
    for i in hor_rectangles:
        cv2.line(abstracted_coral, (i[0],int(i[1]+i[3]/2)),(i[0]+i[2],int(i[1]+i[3]/2)), (255,255,255), 15)


    ver_contours, _ = cv2.findContours(vertical,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    ver_rectangles=[]
    for i in ver_contours:
        if cv2.contourArea(i) > 20:
            ver_rectangles.append(cv2.boundingRect(i))
            temp.append(i)
    ver_contours = temp

    for i in ver_rectangles:
        cv2.line(abstracted_coral, (i[0]+int(i[2]/2),i[1]),(i[0]+int(i[2]/2),i[1]+i[3]), (255,255,255), 15)
    return abstracted_coral, hor_rectangles, ver_rectangles

def getPoints(hor_rectangles,ver_rectangles):
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
    if(x1_of_max_y < x2_of_max_y):
        temp = x1_of_max_y
        x1_of_max_y = x2_of_max_y
        x2_of_max_y = temp
    return x1_of_max_y, x2_of_max_y, desired_y

def process_colar(image):
    

    #########################################    Color Mask    ###########################################
    white_mask, pink_mask = getMask(image)         
    mask = white_mask + pink_mask                                                          
    ######################################################################################################

    ############ Getting contour arround the whole coral and calculate the erosion Constant ##############
    coral_width = getErosionValue(mask)
    ######################################################################################################

    #######################   Applying Morphological Transformations   ###################################

    # 1- Closing Operation to remove small holes in the image (black holes in the mask) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    # edges = cv2.Canny(pink_mask,100, 255)

    # 2- Erosion to flaten the surface of the coral 
    #    the surface isn't smooth in the linage between two pipes 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(coral_width//15,1))
    horizontal = cv2.erode(mask,kernel)               # horizontal Erosion to flaten Vertical surface 
    horizontal_white = cv2.erode(white_mask, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,coral_width//14))
    vertical = cv2.erode(mask,kernel)                 # Vertical Erosion to flaten horizontal surface 
    vertical_white = cv2.erode(white_mask, kernel)
    ######################################################################################################


    ############################### Drawing Abstracted Coral #############################################
    abstracted_coral, hor_rectangles, ver_rectangles = drawAbstractedCoral(image, vertical, horizontal)
    abstracted_coral_white, hor_rectangles_white, ver_rectangles_white = drawAbstractedCoral(image, vertical_white, horizontal_white)
    ### test ###
    # cv2.imshow("hor",image)
    # cv2.imshow("ver",vertical)
    # cv2.imshow("temp",original)
    # cv2.imshow("mask",mask)
    # cv2.imshow("abstract",abstracted_coral)
    # cv2.waitKey(0)
    ######################################################################################################

    ################################# Getting the points #################################################
    x1_of_max_y, x2_of_max_y, desired_y = getPoints(hor_rectangles,ver_rectangles)
    return abstracted_coral, abstracted_coral_white, x1_of_max_y, x2_of_max_y, desired_y
    ######################################################################################################



def main():
    image = cv2.imread("destination.PNG")
    abstracted_coral1, abstracted_coral_white1, abstracted_coral1_x1, abstracted_coral1_x2, abstracted_coral1_y = process_colar(image)
    abstracted_coral1_distance = abs(abstracted_coral1_x1 - abstracted_coral1_x2)
    y, x, _ = image.shape
    # cv2.circle(abstracted_coral1,(abstracted_coral1_x1,abstracted_coral1_y),3,(0,0,255),2)
    # cv2.circle(abstracted_coral1,(abstracted_coral1_x2,abstracted_coral1_y),3,(0,0,255),2)

    tx = 500 - 1 * abstracted_coral1_x1
    ty = 500 - 1 * abstracted_coral1_y
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    img_translationX = cv2.warpAffine(abstracted_coral1, T, (1000, 1000))
    img_translationX_white = cv2.warpAffine(abstracted_coral_white1, T, (1000, 1000))

    image = cv2.imread("2.png")
    abstracted_coral2, abstracted_coral_white2, abstracted_coral2_x1, abstracted_coral2_x2, abstracted_coral2_y = process_colar(image)
    abstracted_coral2_distance = abs(abstracted_coral2_x1 - abstracted_coral2_x2)
    print(abstracted_coral1_distance,abstracted_coral2_distance)
    # cv2.circle(abstracted_coral2,(abstracted_coral2_x1,abstracted_coral2_y),3,(0,0,255),2)
    # cv2.circle(abstracted_coral2,(abstracted_coral2_x2,abstracted_coral2_y),3,(0,0,255),2)

    scale_factor = abstracted_coral1_distance/abstracted_coral2_distance
    height, width, _ = abstracted_coral2.shape
    scaled_abstracted_coral2 = cv2.resize(abstracted_coral2,(int(scale_factor*width),int(scale_factor*height)))
    _,scaled_abstracted_coral2 = cv2.threshold(scaled_abstracted_coral2,10,255,cv2.THRESH_BINARY)

    scaled_abstracted_coral_white2 = cv2.resize(abstracted_coral_white2,(int(scale_factor*width),int(scale_factor*height)))
    _,scaled_abstracted_coral_white2 = cv2.threshold(scaled_abstracted_coral_white2,10,255,cv2.THRESH_BINARY)


    tx = 500 - scale_factor * abstracted_coral2_x1
    ty = 500 - scale_factor * abstracted_coral2_y
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    img_translation = cv2.warpAffine(scaled_abstracted_coral2, T, (1000, 1000))
    img_translation_white = cv2.warpAffine(scaled_abstracted_coral_white2, T, (1000, 1000))

    
    # result = abstracted_coral1 - img_translation

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
    result = cv2.erode(img_translation - img_translationX ,kernel)
    result_white = cv2.erode(img_translation_white - img_translationX_white - result ,kernel)
    # cv2.imshow("destination",abstracted_coral1)
    # cv2.imshow("transformed",img_translation_white)
    # cv2.imshow("result", result)
    cv2.imshow("result White", result_white)
    # cv2.imshow("WMASK OLD", abstracted_coral_white1)
    # cv2.imshow("WMASK NEW", abstracted_coral_white2)
    # cv2.imshow("result",result)
    cv2.waitKey(0)

    img = cv2.warpAffine(image, T, (1000, 1000))
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    binary = cv2.cvtColor(result_white, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary,10,255,cv2.THRESH_BINARY)

    print(binary.shape)
    cott, _= cv2.findContours(binary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cott, -1, (0,255,0), 3)
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    resultsRec = []
    for a in cott :
        resultsRec.append(cv2.boundingRect(a))
    print(resultsRec)
    x,y,w,h = resultsRec[1]
    cv2.rectangle(img,(y, x) , (h, w), (0,0,255), 5 )
    cv2.imshow("Original", img)
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
