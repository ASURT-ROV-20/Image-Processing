import cv2
import numpy as np
import math

# video = cv2.VideoCapture('udpsrc port=8000 ! application/x-rtp,encoding-name=JPEG ! rtpjpegdepay ! jpegdec ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)


def main():
    global frame
    initial_state = True
    frame = cv2.imread('img_02.png')
    distance_threshold = 30
    orientation_threshold = 3
    while True:
        # ret, frame = video.read()
        cnt1, cnt2 = get_blue_line_contours(frame)
        distance = get_distance(cnt1, cnt2)
        angles = get_orientation(cnt1, cnt2)
        draw_contours(cnt1, cnt2)
        if initial_state:
            initial_state = False
            initial_distance = distance
            initial_angles = angles
        else:
            distance_error = initial_distance - distance
            if distance_error < -distance_threshold:
                print(f"negative error -> etla3 fo2 {distance_error}")
            elif distance_error > distance_threshold:
                print(f"positive error -> enzl ta7t {distance_error}")
            else:
                print(f"dont change z yasta")

            orientation_error = initial_angles[1] - angles[1]
            if orientation_error > orientation_threshold:
                print(f"lef ymeen {angles[1]}, {initial_angles[1]}, {orientation_error}")
            elif orientation_error < -orientation_threshold:
                print(f"lef shmal {angles[1]}, {initial_angles[1]}, {orientation_error}")
            else:
                print(f"dont change orientation yasta")
        frame = cv2.imread('img_01.png')

    # video.release()
    cv2.destroyAllWindows()


def get_blue_line_contours(frame):
    frame = cv2.resize(frame, (600, 400))
    thresh = get_blue_mask(frame)
    return get_largest_two_contours(thresh)


def draw_contours(cnt1, cnt2):
    mask = np.zeros(frame.shape, np.uint8)
    cv2.drawContours(mask, [cnt1, cnt2], -1, (0, 255, 0), 1)
    cv2.imshow("test", mask)
    cv2.waitKey()


def get_blue_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    return thresh


def get_largest_two_contours(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if get_vertex_count(c) == 2]
    largest_two_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    return largest_two_contours


def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
    return len(approx)


def get_distance(cnt1, cnt2):
    m1 = midpoint(cnt1)
    m2 = midpoint(cnt2)
    distance = math.sqrt(pow(m1[0] - m2[0], 2) + pow(m1[1] - m2[1], 2))
    return distance


def midpoint(cnt):
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    mid = (bottommost[0] + topmost[0]) / 2, (bottommost[1] + topmost[1]) / 2
    return mid


def get_orientation(cnt1, cnt2):
    return get_orientation_from_contour(cnt1), get_orientation_from_contour(cnt2)


def get_orientation_from_contour(cnt1):
    _, (width, height), orientation = cv2.minAreaRect(cnt1)
    # make the angle in the [0, 180) range *-ve
    if width < height:
        orientation = orientation - 90
    return abs(orientation)


if __name__ == '__main__':
    main()
