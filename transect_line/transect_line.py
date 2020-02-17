import cv2
import numpy as np
import math
# import rospy
# from geometry_msgs.msg import Quaternion

# video = cv2.VideoCapture('udpsrc port=8000 ! application/x-rtp,encoding-name=JPEG ! rtpjpegdepay ! jpegdec ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
video = cv2.VideoCapture('http://192.168.1.151:8080/video')
video = cv2.VideoCapture('my_vid.mp4')


# def init_ros():
#     global rate
#     rospy.init_node('autonomous_node', anonymous=True)
#     rate = rospy.Rate(10)  # 10hz  # ? rospy rate?
#     # rospy.init_node('qt_equation_listener', anonymous=True)
#     # todo send ros msgs in a new thread
#
#
# def create_ros_publisher():
#     return rospy.Publisher("rov_velocity", Quaternion, queue_size=10)  # ? rospy queue size?


def main():
    global frame
    initial_state = True
    # init_ros()
    # publisher = create_ros_publisher()
    # success, frame = video.read()
    success = True
    distance_threshold = 30
    orientation_threshold = 3
    while success:
        success, frame = video.read()
        cnt1and2 = get_blue_line_contours(frame)
        if cnt1and2 is None or len(cnt1and2) < 2:
            print(f"error: no blue lines found")
            continue
        else:
            cnt1, cnt2 = cnt1and2

        distance = get_distance(cnt1, cnt2)
        angles = get_orientation(cnt1, cnt2)
        draw_contours(cnt1, cnt2)
        if initial_state:
            initial_state = False
            initial_distance = distance
            initial_angles = 90
        else:
            distance_error = initial_distance - distance

            up = ""
            right = ""
            if distance_error < -distance_threshold:
                publish_direction()
                # print(f"negative error -> etla3 fo2 {distance_error}")
                up = "uppp " + str(distance_error)

            elif distance_error > distance_threshold:
                up = "down " + str(distance_error)
            else:
                up = "None"

            orientation_error = initial_angles - angles[1]
            if orientation_error > orientation_threshold:
                right = "right " + str(orientation_error)
                # print(f"lef ymeen {angles[1]}, {initial_angles}, {orientation_error}")
            elif orientation_error < -orientation_threshold:
                right = "left " + str(orientation_error)
                # print(f"lef shmal {angles[1]}, {initial_angles}, {orientation_error}")
            else:
                right = "None "
                # print(f"dont change orientation yasta")
            print(up, right)
        frame = cv2.imread('img_01.png')
        # rate.sleep() # ros
    video.release()
    cv2.destroyAllWindows()


def publish_direction(z, w):
    pass
    # publisher.publish(motion_json)
    # geometry_msgs:
    # :Quaternion
    # msg;
    # msg.x = prevX;
    # msg.y = prevY;
    # msg.z = prevZ
    # msg.w = prevR
    # publisher.publish(msg);


def get_blue_line_contours(frame):
    frame = cv2.resize(frame, (600, 400))
    cv2.imshow("frame", frame)
    thresh = get_blue_mask(frame)
    return get_largest_two_contours(thresh)


def draw_contours(cnt1, cnt2):
    mask = np.zeros(frame.shape, np.uint8)
    cv2.drawContours(mask, [cnt1, cnt2], -1, (0, 255, 0), 1)
    cv2.imshow("test", mask)
    cv2.waitKey(2)


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
