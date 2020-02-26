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


## abstracted : abs, coordinate: coord # to avoid long names :')
def main():
    old_image = cv2.imread("destination.PNG")
    absOldCoral, absOldCoral_pinkMask, absOldCoral_firstRoot_xCoord,\
     absOldCoral_secondRoot_xCoord, absOldCoral_roots_yCoord = process_colar(old_image)
    
    absOldCoral_distance = abs(absOldCoral_firstRoot_xCoord - absOldCoral_secondRoot_xCoord)
    y, x, _ = old_image.shape

    tx = 500 - 1 * absOldCoral_firstRoot_xCoord
    ty = 500 - 1 * absOldCoral_roots_yCoord
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    # translated: trans
    trans_absOldCoral = cv2.warpAffine(absOldCoral, T, (1000, 1000))
    trans_absOldCoral_pinkMask = cv2.warpAffine(absOldCoral_pinkMask, T, (1000, 1000))

    image = cv2.imread("2.png")
    absNewCoral, absNewCoral_pinkMask, absNewCoral_firstRoot_xCoord,\
     absNewCoral_secondtRoot_xCoord, absNewCoral_roots_yCoord = process_colar(image)

    absNewCoral_distance = abs(absNewCoral_firstRoot_xCoord - absNewCoral_secondtRoot_xCoord)
    print(absOldCoral_distance,absNewCoral_distance)

    scaleFactor = absOldCoral_distance/absNewCoral_distance
    height, width, _ = absNewCoral.shape
    scaled_absNewCoral = cv2.resize(absNewCoral,(int(scaleFactor*width),int(scaleFactor*height)))
    _,scaled_absNewCoral = cv2.threshold(scaled_absNewCoral,10,255,cv2.THRESH_BINARY)

    scaled_absNewCoral_pinkMask = cv2.resize(absNewCoral_pinkMask,(int(scaleFactor*width),int(scaleFactor*height)))
    _,scaled_absNewCoral_pinkMask = cv2.threshold(scaled_absNewCoral_pinkMask,10,255,cv2.THRESH_BINARY)


    tx = 500 - scaleFactor * absNewCoral_firstRoot_xCoord
    ty = 500 - scaleFactor * absNewCoral_roots_yCoord
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    trans_absNewCoral = cv2.warpAffine(scaled_absNewCoral, T, (1000, 1000))
    trans_absNewCoral_pinkMask = cv2.warpAffine(scaled_absNewCoral_pinkMask, T, (1000, 1000))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
    growthResult = cv2.erode(trans_absNewCoral - trans_absOldCoral ,kernel)
    deathResult = cv2.erode(trans_absOldCoral - trans_absNewCoral ,kernel)
    recoveryResult = cv2.erode(trans_absNewCoral_pinkMask - trans_absOldCoral_pinkMask - growthResult ,kernel)
    bleachingResult = cv2.erode(trans_absOldCoral_pinkMask - trans_absNewCoral_pinkMask - deathResult ,kernel)

    resultImage = cv2.resize(image,(int(scaleFactor*width),int(scaleFactor*height)))
    resultImage = cv2.warpAffine(resultImage, T, (1000, 1000))
    cv2.imshow("Original", resultImage)
    cv2.waitKey(0)

    # death results
    _, deathBinary = cv2.threshold(cv2.cvtColor(growthResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    deathContours, _= cv2.findContours(deathBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, deathContours, -1, (125,125,125), 3)
    deathResultsRec = []
    for i in deathContours :
        deathResultsRec.append(cv2.boundingRect(i))
    print(deathResultsRec)
    for i in range(len(deathResultsRec)):
        cv2.rectangle(resultImage, (int(deathResultsRec[i][0]), int(deathResultsRec[i][1])), \
            (int(deathResultsRec[i][0]+deathResultsRec[i][2]), int(deathResultsRec[i][1]+deathResultsRec[i][3])), (0,255,255), 2)
    # growth results
    _, growthBinary = cv2.threshold(cv2.cvtColor(growthResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    growthContours, _= cv2.findContours(growthBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, growthContours, -1, (125,125,125), 3)
    growthResultsRec = []
    for i in growthContours :
        growthResultsRec.append(cv2.boundingRect(i))
    print(growthResultsRec)
    for i in range(len(growthResultsRec)):
        cv2.rectangle(resultImage, (int(growthResultsRec[i][0]), int(growthResultsRec[i][1])), \
            (int(growthResultsRec[i][0]+growthResultsRec[i][2]), int(growthResultsRec[i][1]+growthResultsRec[i][3])), (0,255,0), 2)
    # recovery results
    _, recoveryBinary = cv2.threshold(cv2.cvtColor(recoveryResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    recoveryContours, _= cv2.findContours(recoveryBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, recoveryContours, -1, (125,125,125), 3)
    recoveryResultsRec = []
    for i in recoveryContours :
        recoveryResultsRec.append(cv2.boundingRect(i))
    print(recoveryResultsRec)
    for i in range(len(recoveryResultsRec)):
        cv2.rectangle(resultImage, (int(recoveryResultsRec[i][0]), int(recoveryResultsRec[i][1])), \
            (int(recoveryResultsRec[i][0]+recoveryResultsRec[i][2]), int(recoveryResultsRec[i][1]+recoveryResultsRec[i][3])), (255,0,0), 2)
    # bleaching results
    _, bleachingBinary = cv2.threshold(cv2.cvtColor(bleachingResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    bleachingContours, _= cv2.findContours(bleachingBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, bleachingContours, -1, (125,125,125), 3)
    bleachingResultsRec = []
    for i in bleachingContours :
        bleachingResultsRec.append(cv2.boundingRect(i))
    print(bleachingResultsRec)
    for i in range(len(bleachingResultsRec)):
        cv2.rectangle(resultImage, (int(bleachingResultsRec[i][0]), int(bleachingResultsRec[i][1])), \
            (int(bleachingResultsRec[i][0]+bleachingResultsRec[i][2]), int(bleachingResultsRec[i][1]+bleachingResultsRec[i][3])), (0,0,255), 2)

    
    cv2.imshow("Original", resultImage)
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

    # cv2.circle(absOldCoral,(absOldCoral_firstRoot_xCoord,absOldCoral_roots_yCoord),3,(0,0,255),2)
    # cv2.circle(absOldCoral,(absOldCoral_secondRoot_xCoord,absOldCoral_roots_yCoord),3,(0,0,255),2)
    # cv2.circle(absNewCoral,(absNewCoral_firstRoot_xCoord,absNewCoral_roots_yCoord),3,(0,0,255),2)
    # cv2.circle(absNewCoral,(absNewCoral_secondtRoot_xCoord,absNewCoral_roots_yCoord),3,(0,0,255),2)
    # cv2.imshow("destination",absOldCoral)
    # cv2.imshow("transformed",trans_absNewCoral_pinkMask)
    # cv2.imshow("result", result)
    # cv2.imshow("WMASK OLD", absOldCoral_pinkMask)
    # cv2.imshow("WMASK NEW", absNewCoral_pinkMask)
    # cv2.imshow("result",result)