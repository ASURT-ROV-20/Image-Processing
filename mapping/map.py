import numpy as np
from equation import equation
import cv2

def main():

    # to calc time
    e1 = cv2.getTickCount()
    
    # read image
    image_path = "map.jpg"
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
   
    # init output photo
    # 40 is a factor to increase image resolution
    output = np.zeros((6*40, 18*40, 3))
    output[output == 0] = 255
    for i in range(4):
        cv2.line(output, (0 ,i * 2 *40), (19 *40, i * 2 *40), (0,0,0), 10)
    for i in range(10):
        cv2.line(output, (i * 2 *40, 0), (i * 2 *40 , 6 *40), (0,0,0), 10)

    # filters
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernal = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernal, iterations=2)
    kernal = np.ones((5,5), np.uint8)
    edges = cv2.erode(edges, kernal, iterations=1)
    # edges = cv2.bilateralFilter(edges,20, 00, 100)
    # cv2.imshow("after filters", edges)
    # cv2.imshow("canny", edges)

    # detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 90, minLineLength= 50 ,maxLineGap=100)

    # delete lines 
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(edges, (x1,y1), (x2, y2), (0,255,0), 7)
    # cv2.imshow("lines", edges)
    img_shape = np.ones(edges.shape)
    edges[img_shape == 0] = 0
    # cv2.imshow("edges deleted", edges)

    
    print("Number of lines = " + str(len(lines)))
    lines2 = reduce_lines(lines)
    print("Number of redued lines = " + str(len(lines2)))

    y_equs , x_equs = classify(lines2)
    for i in range(len(y_equs)):
        x1, y1, x2, y2 =  y_equs[i].info()
        # print(x1, y1, x2, y2)
        # cv2.line(image, (x1,y1), (x2, y2), (0,255,0), 3)
    for i in range(len(x_equs)):
        x1, y1, x2, y2 =  x_equs[i].info()
        # print(x1, y1, x2, y2)
        # cv2.line(image, (x1,y1), (x2, y2), (0,255,0), 3)
    # cv2.imshow("edges deleted", image)
    print("Number of Horizontal lines = " + str(len(y_equs)))
    print("Number of Vertical lines = " + str(len(x_equs)))

     
    ## find grids by colors
    # convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ## yellow squares
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    yellow_img = cv2.bitwise_and(image, image, mask= mask_yellow)
    # cv2.imshow("after mask", yellow_img)
    # find contours
    yellow_img = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(yellow_img, 1, 2)
    print("Number of squares = " + str(len(contours)))
    # for i in range(len(contours)):
    #     cv2.drawContours(image, contours, i, (0,0,255), 2)
    # TODO
    # compaere with 90% from points
    for contour in contours :
        x = contour[0][0][0]
        y = contour[0][0][1]
        x_grid = 0
        y_grid = 0
        for i in range(len(y_equs) - 1) :
            if y > y_equs[i].y_equal(x) and y < y_equs[i+1].y_equal(x) :
                break
            y_grid += 1
        for i in range(len(x_equs) - 1) :
            if x > x_equs[i].x_equal(y) and x < x_equs[i+1].x_equal(y) :
                break
            x_grid += 1
        cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (0,255,255), 20)
        print( " x = " + str(x_grid) + "  y = " + str(y_grid) + "  shape: Square" )

    ## red stars
    lower_red = np.array([150,100,100])
    upper_red = np.array([179,255,255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    red_img = cv2.bitwise_and(image, image, mask= mask_red)
    # cv2.imshow("after mask", red_img)
  
    red_img = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(red_img, 1, 2)
    print("Number of stars = " + str(len(contours)))
    # for i in range(len(contours)):
    #     cv2.drawContours(image, contours, i, (0,0,255), 2)
    for contour in contours :
        x = contour[0][0][0]
        y = contour[0][0][1]
        x_grid = 0
        y_grid = 0
        for i in range(len(y_equs) - 1) :
            if y > y_equs[i].y_equal(x) and y < y_equs[i+1].y_equal(x) :
                break
            y_grid += 1
        for i in range(len(x_equs) - 1) :
            if x > x_equs[i].x_equal(y) and x < x_equs[i+1].x_equal(y) :
                break
            x_grid += 1
        cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (255,0,0), 20)
        print( " x = " + str(x_grid) + "  y = " + str(y_grid) + "  shape: Star" )

    ## black clony
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    black_img = cv2.bitwise_and(image, image, mask= mask_black)
    
  
    black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("after mask", black_img)
    contours, _ = cv2.findContours(black_img, 1, 2)
    print("Number of clonies = " + str(len(contours)))
    # for i in range(len(contours)):
    #     cv2.drawContours(image, contours, i, (0,0,255), 2)
    for contour in contours :
        x = contour[0][0][0]
        y = contour[0][0][1]
        x_grid = 0
        y_grid = 0
        for i in range(len(y_equs) - 1) :
            if y > y_equs[i].y_equal(x) and y < y_equs[i+1].y_equal(x) :
                break
            y_grid += 1
        for i in range(len(x_equs) - 1) :
            if x > x_equs[i].x_equal(y) and x < x_equs[i+1].x_equal(y) :
                break
            x_grid += 1
        # cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (0,0,255), 20)
        print( " x = " + str(x_grid) + "  y = " + str(y_grid) + "  shape: clony" )

    ## white circle
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    white_img = cv2.bitwise_and(image, image, mask= mask_white)
    # cv2.imshow("after mask", white_img)
  
    white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(white_img, 1, 2)
    print("Number of circles = " + str(len(contours)))
    # for i in range(len(contours)):
    #     cv2.drawContours(image, contours, i, (0,0,255), 2)
    for contour in contours :
        x = contour[0][0][0]
        y = contour[0][0][1]
        x_grid = 0
        y_grid = 0
        for i in range(len(y_equs) - 1) :
            if y > y_equs[i].y_equal(x) and y < y_equs[i+1].y_equal(x) :
                break
            y_grid += 1
        for i in range(len(x_equs) - 1) :
            if x > x_equs[i].x_equal(y) and x < x_equs[i+1].x_equal(y) :
                break
            x_grid += 1
        # cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (0,255,0), 20)
        print( " x = " + str(x_grid) + "  y = " + str(y_grid) + "  shape: circle" )

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print (time)
    # output = cv2.resize(output, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow("Img", edges)
    # cv2.waitKey(0)
    
def reduce_lines(lines):
    max_tolorence = 1
    max_distance = 55
    lines_equs = []
    for line in lines:
        lines_equs.append(equation(line[0][0], line[0][1], line[0][2], line[0][3]))
    lines_equs = sorted(lines_equs, key= lambda x : x.length(), reverse=True)
    for line1 in lines_equs:
        for line2 in lines_equs:
            if line1 != line2 and abs(line1.slope - line2.slope) < max_tolorence :
                if(line1.distance_from(line2) < max_distance):
                    lines_equs.remove(line2)
    for line1 in lines_equs:
        for line2 in lines_equs:      
            if line1 != line2 and abs(line1.rho - line2.rho) < 30 and abs(line1.theta - line2.theta) < 1 :
                lines_equs.remove(line2)
    return lines_equs


def classify(lines):
    Hequs = []
    Vequs = []
    for line in lines:
        if line.is_vertical():
            Vequs.append(line)
        else:
            Hequs.append(line)
    Hequs = sorted(Hequs, key= lambda x : x.y1)
    Vequs = sorted(Vequs, key= lambda x : x.x1)
    return Hequs, Vequs
    


if __name__ == "__main__":
    main()