import numpy as np
import cv2

def main():

    # to calc time
    e1 = cv2.getTickCount()
    
    # read image
    image_path = "map.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=280)
    
    print("Number of lines = " + str(len(lines)))
    lines2 = reduce_lines(lines)
    horizontal, vertical = classify(lines2)
    print("Number of Horizontal lines = " + str(len(horizontal)))
    print("Number of Vertical lines = " + str(len(vertical)))

    # delete lines 
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(edges, (x1,y1), (x2, y2), (0,255,0), 3)
    img_shape = np.ones(edges.shape)
    edges[img_shape == 0] = 0

    # find contours
    contours, hierarchy = cv2.findContours(edges, 1, 2)
    print("Number of shapes = " + str(len(contours)))
    for i in range(len(contours)):
        cv2.drawContours(image, contours, i, (0,0,255), 2)


    # init output photo
    # 40 is a factor to increase image resolution
    output = np.zeros((6*40, 18*40, 3))
    output[output == 0] = 255
    for i in range(4):
        cv2.line(output, (0 ,i * 2 *40), (19 *40, i * 2 *40), (0,0,0), 10)
    for i in range(10):
        cv2.line(output, (i * 2 *40, 0), (i * 2 *40 , 6 *40), (0,0,0), 10)


    # approximate polggon and find grids
    # TODO
    # compaere with 90% from points
    y_equs , x_equs = find_mean(horizontal, vertical)
    for contour in contours :
        x = contour[0][0][0]
        y = contour[0][0][1]
        x_grid = 0
        y_grid = 0
        for i in range(len(y_equs) - 1) :
            if y > y_equs[i] and y < y_equs[i+1] :
                break
            y_grid += 1
        for i in range(len(x_equs) - 1) :
            if x > x_equs[i] and x < x_equs[i+1] :
                break
            x_grid += 1

        ## TODO
        ## find len(approx) && 0.04
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) > 8 :
            cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (255,0,0), 20)
        if len(approx) == 4 :
            cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (0,255,255), 20)
        if len(approx) == 3 :
            cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (255,0,0), 20)
        if len(approx) == 7 :
            cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (0,255,0), 20)
        if len(approx) == 6 :
            cv2.circle(output, (x_grid*80 + 40, y_grid*80 + 40), 10, (0,0,255), 20)
    
        print( " x = " + str(x_grid) + "  y = " + str(y_grid) + "  lens = " + str(len(approx)))

    

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print (time)
    # output = cv2.resize(output, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    # cv2.imshow("Img", edges)
    # cv2.waitKey(0)
    
def reduce_lines(lines):
    max_distance = 10
    lines = lines.tolist()
    for line1 in lines:
        for line2 in lines:
            if line1 != line2 and abs(line1[0][0] - line2[0][0]) < max_distance and abs(line1[0][1] - line2[0][1]) < max_distance \
                and abs(line1[0][2] - line2[0][2]) < max_distance and abs(line1[0][3] - line2[0][3]) < max_distance :
                # np.delete(lines, np.argwhere(lines == line2))
                lines.remove(line2)
    return lines


def classify(lines):
    Hlines = []
    Vlines = []
    max_distance = 50
    for line in lines:
        if abs(line[0][0] - line[0][2]) < max_distance :
            Vlines.append(line)
        if abs(line[0][1] - line[0][3]) < max_distance :
            Hlines.append(line)
    return Hlines, Vlines

def find_mean(Hlines, Vlines):
    Hmean = []
    Vmean = []
    for line in Vlines:
        Vmean.append((line[0][0] + line[0][2]) / 2)
    for line in Hlines:
        Hmean.append((line[0][1] + line[0][3]) / 2)
    Hmean = sorted(Hmean)
    Vmean = sorted(Vmean)
    return Hmean, Vmean


if __name__ == "__main__":
    main()