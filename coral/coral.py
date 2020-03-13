import numpy as np
import cv2
import math
import random

def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

def getMask(image):
    hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)                      
    lower_pink= np.array([150,80,50])
    higher_pink = np.array([170,255,255])
    pink_mask= cv2.inRange(hsv,lower_pink,higher_pink)

    lower_white= np.array([0,0,200])
    higher_white = np.array([255,70,255])
    white_mask= cv2.inRange(hsv,lower_white,higher_white)

    cv2.imshow("mask",pink_mask+white_mask)
    cv2.waitKey(0)

    return pink_mask, white_mask        

def getErosionValue(mask):
    cott, _ = cv2.findContours(mask,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

    contours_Dic = {}
    for i in range(len(cott)):
        area = cv2.contourArea(cott[i])
        contours_Dic[area] = i

    i = contours_Dic[max(list(contours_Dic))]
    
    _, _, coral_width, _ = cv2.boundingRect(cott[i])
    return coral_width // 15

def drawAbstractedCoral(image, vertical, horizontal, erosiosFactor):
    erosiosFactor = 0
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
        cv2.line(abstracted_coral, (i[0] - erosiosFactor,int(i[1]+i[3]/2)),(i[0]+i[2]+erosiosFactor,int(i[1]+i[3]/2)), (255,255,255), 30)


    ver_contours, _ = cv2.findContours(vertical,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    ver_rectangles=[]
    for i in ver_contours:
        if cv2.contourArea(i) > 20:
            ver_rectangles.append(cv2.boundingRect(i))
            temp.append(i)
    ver_contours = temp

    # cv2.imshow("hor",abstracted_coral)
    # cv2.imshow("ver",vertical)
    # cv2.waitKey(0)

    cv2.waitKey(0)
    for i in ver_rectangles:
        cv2.line(abstracted_coral, (i[0]+int(i[2]/2),i[1]-erosiosFactor),(i[0]+int(i[2]/2),i[1]+i[3]+erosiosFactor), (255,255,255), 30)
    return abstracted_coral, hor_rectangles, ver_rectangles


def getPoints(hor_rectangles,ver_rectangles):
    max_y=0
    for i in hor_rectangles:
        if( int(i[1]+i[3]/2) > max_y):
            max_y = int(i[1]+i[3]/2)
            x_of_max_y = i[0]
    #cv2.circle(abstracted_coral,(x_of_max_y,max_y),3,(0,0,255),2) # test
    desired_y = max_y
    index = -1
    max_y=0
    x1_of_max_y=0
    x2_of_max_y=0
    reversed_ver_rectangles = list(reversed(ver_rectangles))

    counter = 0

    for i in ver_rectangles:
        if( (i[1]+i[3]) > max_y):
            max_y = i[1]+i[3]
            x1_of_max_y = i[0]+int(i[2]/2)
            index = counter
        counter +=1
    #cv2.circle(abstracted_coral,(x1_of_max_y,max_y),3,(0,0,255),2) # test
    print("x1",x1_of_max_y,"y",max_y)
    del ver_rectangles[index]

    max_y=0
    for i in ver_rectangles:
        if( (i[1]+i[3]) > max_y):
            max_y = i[1]+i[3]
            x2_of_max_y = i[0]+int(i[2]/2)
    print("x2",x2_of_max_y,"y",max_y)
    
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
    erosiosFactor = getErosionValue(mask)
    ######################################################################################################

    #######################   Applying Morphological Transformations   ###################################

    # 1- Closing Operation to remove small holes in the image (black holes in the mask) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    # edges = cv2.Canny(pink_mask,100, 255)

    # 2- Erosion to flaten the surface of the coral 
    #    the surface isn't smooth in the linage between two pipes 
    print("erosion factor:", erosiosFactor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(erosiosFactor,3))
    horizontal = cv2.erode(mask,kernel)               # horizontal Erosion to flaten Vertical surface 
    horizontal_white = cv2.erode(white_mask, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,erosiosFactor))
    vertical = cv2.erode(mask,kernel)                 # Vertical Erosion to flaten horizontal surface 
    vertical_white = cv2.erode(white_mask, kernel)
    ######################################################################################################


    ############################### Drawing Abstracted Coral #############################################
    abstracted_coral, hor_rectangles, ver_rectangles = drawAbstractedCoral(image, vertical, horizontal, erosiosFactor)
    abstracted_coral_white, hor_rectangles_white, ver_rectangles_white = drawAbstractedCoral(image, vertical_white, horizontal_white,erosiosFactor)
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
    oldImage = cv2.imread("destination.png")
    # oldImage = cv2.resize(oldImage,(500,500))
    absOldCoral, absOldCoral_pinkMask, absOldCoral_firstRoot_xCoord,\
     absOldCoral_secondRoot_xCoord, absOldCoral_roots_yCoord = process_colar(oldImage)
    
    absOldCoral_distance = abs(absOldCoral_firstRoot_xCoord - absOldCoral_secondRoot_xCoord)
    y, x, _ = oldImage.shape

    tx = 1500 - 1 * absOldCoral_firstRoot_xCoord
    ty = 1000 - 1 * absOldCoral_roots_yCoord
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    # translated: trans
    trans_absOldCoral = cv2.warpAffine(absOldCoral, T, (3000, 3000))
    trans_absOldCoral_pinkMask = cv2.warpAffine(absOldCoral_pinkMask, T, (3000, 3000))

    image = cv2.imread("new.jpeg")
    image = simplest_cb(image)
    
    cv2.imshow("color corrected", image)
    cv2.waitKey(0)

    # image = cv2.resize(image,(500,500))
    absNewCoral, absNewCoral_pinkMask, absNewCoral_firstRoot_xCoord,\
     absNewCoral_secondtRoot_xCoord, absNewCoral_roots_yCoord = process_colar(image)

    absNewCoral_distance = abs(absNewCoral_firstRoot_xCoord - absNewCoral_secondtRoot_xCoord)
    print(absNewCoral_firstRoot_xCoord , absNewCoral_secondtRoot_xCoord)
    print(absOldCoral_distance,absNewCoral_distance)
    ###############
    # cv2.circle(absNewCoral,(absNewCoral_firstRoot_xCoord,absNewCoral_roots_yCoord),3,(0,0,255),2)
    # cv2.circle(absNewCoral,(absNewCoral_secondtRoot_xCoord,absNewCoral_roots_yCoord),3,(0,0,255),2)

    # cv2.circle(absOldCoral,(absOldCoral_firstRoot_xCoord,absOldCoral_roots_yCoord),3,(0,0,255),2)
    # cv2.circle(absOldCoral,(absOldCoral_secondRoot_xCoord,absOldCoral_roots_yCoord),3,(0,0,255),2)

    # cv2.imshow("new",absNewCoral)
    # cv2.imshow("old",absOldCoral)
    # cv2.waitKey(0)
    ##################

    scaleFactor = absOldCoral_distance/absNewCoral_distance
    height, width, _ = absNewCoral.shape
    scaled_absNewCoral = cv2.resize(absNewCoral,(int(scaleFactor*width),int(scaleFactor*height)))
    _,scaled_absNewCoral = cv2.threshold(scaled_absNewCoral,10,255,cv2.THRESH_BINARY)

    scaled_absNewCoral_pinkMask = cv2.resize(absNewCoral_pinkMask,(int(scaleFactor*width),int(scaleFactor*height)))
    _,scaled_absNewCoral_pinkMask = cv2.threshold(scaled_absNewCoral_pinkMask,10,255,cv2.THRESH_BINARY)



    tx = 1500 - scaleFactor * absNewCoral_firstRoot_xCoord
    ty = 1000 - scaleFactor * absNewCoral_roots_yCoord
    T = np.float32([[1, 0, tx], [0, 1, ty]]) 

    trans_absNewCoral = cv2.warpAffine(scaled_absNewCoral, T, (3000, 3000))
    trans_absNewCoral_pinkMask = cv2.warpAffine(scaled_absNewCoral_pinkMask, T, (3000, 3000))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    growthResult = cv2.erode(trans_absNewCoral - trans_absOldCoral ,kernel)
    deathResult = cv2.erode(trans_absOldCoral - trans_absNewCoral ,kernel)
    recoveryResult = cv2.erode(trans_absNewCoral_pinkMask - trans_absOldCoral_pinkMask - growthResult ,kernel)
    bleachingResult = cv2.erode(trans_absOldCoral_pinkMask - trans_absNewCoral_pinkMask - deathResult ,kernel)


    resultImage = cv2.resize(image,(int(scaleFactor*width),int(scaleFactor*height)))
    resultImage = cv2.warpAffine(resultImage, T, (4000, 4000))
    # cv2.imshow("Original", resultImage)
    cv2.waitKey(0)

    # death results
    _, deathBinary = cv2.threshold(cv2.cvtColor(growthResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    deathContours, _= cv2.findContours(deathBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, deathContours, -1, (125,125,125), 3)
    deathResultsRec = []
    for i in deathContours :
        if(cv2.contourArea(i)>300):
            deathResultsRec.append(cv2.boundingRect(i))
    for i in range(len(deathResultsRec)):
        cv2.rectangle(resultImage, (int(deathResultsRec[i][0]), int(deathResultsRec[i][1])), \
            (int(deathResultsRec[i][0]+deathResultsRec[i][2]), int(deathResultsRec[i][1]+deathResultsRec[i][3])), (0,255,255), 2)
    # growth results
    _, growthBinary = cv2.threshold(cv2.cvtColor(growthResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    growthContours, _= cv2.findContours(growthBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, growthContours, -1, (125,125,125), 3)
    growthResultsRec = []
    for i in growthContours :
        if(cv2.contourArea(i)>300):
            growthResultsRec.append(cv2.boundingRect(i))
    for i in range(len(growthResultsRec)):
        cv2.rectangle(resultImage, (int(growthResultsRec[i][0]), int(growthResultsRec[i][1])), \
            (int(growthResultsRec[i][0]+growthResultsRec[i][2]), int(growthResultsRec[i][1]+growthResultsRec[i][3])), (0,255,0), 2)
    # recovery results
    _, recoveryBinary = cv2.threshold(cv2.cvtColor(recoveryResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    recoveryContours, _= cv2.findContours(recoveryBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, recoveryContours, -1, (125,125,125), 3)
    recoveryResultsRec = []
    for i in recoveryContours :
        if(cv2.contourArea(i)>300):
            recoveryResultsRec.append(cv2.boundingRect(i))
    for i in range(len(recoveryResultsRec)):
        cv2.rectangle(resultImage, (int(recoveryResultsRec[i][0]), int(recoveryResultsRec[i][1])), \
            (int(recoveryResultsRec[i][0]+recoveryResultsRec[i][2]), int(recoveryResultsRec[i][1]+recoveryResultsRec[i][3])), (255,0,0), 2)
    # bleaching results
    _, bleachingBinary = cv2.threshold(cv2.cvtColor(bleachingResult, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    bleachingContours, _= cv2.findContours(bleachingBinary,mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultImage, bleachingContours, -1, (125,125,125), 3)
    bleachingResultsRec = []
    for i in bleachingContours :
        if(cv2.contourArea(i)>300):
            bleachingResultsRec.append(cv2.boundingRect(i))
    for i in range(len(bleachingResultsRec)):
        cv2.rectangle(resultImage, (int(bleachingResultsRec[i][0]), int(bleachingResultsRec[i][1])), \
            (int(bleachingResultsRec[i][0]+bleachingResultsRec[i][2]), int(bleachingResultsRec[i][1]+bleachingResultsRec[i][3])), (0,0,255), 2)

    trans_absNewCoral = cv2.resize(trans_absNewCoral,(2400,2400))
    trans_absOldCoral = cv2.resize(trans_absOldCoral,(2400,2400))
    growthResult = cv2.resize(growthResult,(2400,2400))
    resultImage = cv2.resize(resultImage,(2400,2400))
    cv2.imshow("new",trans_absNewCoral)
    cv2.imshow("old",trans_absOldCoral)
    cv2.imshow("result",growthResult)
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