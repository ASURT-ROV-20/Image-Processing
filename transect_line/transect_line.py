import cv2
import numpy as np
# todo determine slope and determine theta
# todo determine xcostheta lletenen
# todo distance  = max _i - x_j
# video = cv2.VideoCapture('udpsrc port=8000 ! application/x-rtp,encoding-name=JPEG ! rtpjpegdepay ! jpegdec ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
initial_state = True
frame = cv2.imread('img_06.png')
distance_threshold = 30
orientation_threshold = 0.87


def get_distance(frame):
    frame = cv2.resize(frame, (600, 400))
    thresh = get_blue_mask(frame)
    edges = cv2.Canny(thresh, 150, 250, 5)
    lines = cv2.HoughLinesP(edges, 2, 1 * np.pi / 180, 45, minLineLength=60, maxLineGap=60)
    mask = np.zeros(frame.shape, np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(len(lines))
    cv2.imshow("test", mask)
    cv2.waitKey()

    largest_two_contours = get_largest_two_contours(lines)
    mask = np.zeros(frame.shape, np.uint8)
    cv2.drawContours(mask, largest_two_contours, -1, (0, 255, 0), 1)
    cv2.imshow("test", mask)
    cv2.waitKey()
    return cv2.boundingRect(largest_two_contours[1])[0] - \
           cv2.boundingRect(largest_two_contours[0])[0]


def get_largest_two_contours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contours = [c for c in contours if get_vertex_count(c) == 4]
    largest_two_contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:3]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        print(f"h is{h}")
        print(f"w is{w}")
        print(f"ratio is{h/w}")
    # x, y, w, h = cv2.boundingRect(c)
    # return (h / w) > 4
    # print(f"#of vertices {get_vertex_count(largest_two_contours)}")
    return largest_two_contours


def get_blue_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    return thresh


def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    return len(approx)


def get_orientation(frame):
    frame = cv2.resize(frame, (600, 400))
    thresh = get_blue_mask(frame)
    cnt1, cnt2 = get_largest_two_contours(thresh)
    rect = cv2.minAreaRect(cnt1)
    orientation1 = rect[-1]
    print(orientation1)
    rect = cv2.minAreaRect(cnt2)
    orientation2 = rect[-1]
    print(orientation2)
    return orientation1, orientation2


print(get_distance(frame))
cv2.waitKey()

while True:
    # ret, frame = video.read()
    distance = get_distance(frame)
    angles = get_orientation(frame)
    if initial_state:
        initial_state = False
        initial_distance = distance
        initial_angles = angles
    else:
        distance_error = distance - initial_distance
        if distance_error > distance_threshold:
            print(f"positive error -> etla3 {distance_error}")
        elif distance_error < -distance_threshold:
            print(f"negative error -> enzl {distance_error}")
        else:
            print(f"emshi odam yasta1")

        orientation_error = angles[0] - initial_angles[0]
        if angles[0] > orientation_threshold:
            print(f"lef shmal {orientation_error}")
        elif angles[0] < -orientation_threshold:
            print(f"lef ymeen {orientation_error}")
        else:
            print(f"emshi odam yasta2")

    frame = cv2.imread('img_06.png')

# video.release()
cv2.destroyAllWindows()
