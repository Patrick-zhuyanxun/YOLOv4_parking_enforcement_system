from turtle import delay
import cv2
import time
import numpy as np


def line_detect(img_or, thresh):

    kernel = np.ones((7, 7), np.uint8)  # 可調mask大小
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)
    cv2.imshow("canny", edges)
    #cv2.imwrite('/media/anson/Disk1/demo/img_test/canny.png', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60,
                            minLineLength=20, maxLineGap=10)  # 可調最小長度,最大間格

    return lines


def filter_red1(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([156, 43, 46])  # 可調hsv,紅色 0-10|156-180/43-255/46-255
    upper_red = np.array([180, 100, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.bitwise_and(img, img, mask=mask)


def filter_red2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.bitwise_and(img, img, mask=mask)


def combine(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    ret, thresh2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY)
    return thresh1 | thresh2

def image(img):
    img_red1 = filter_red1(img)
    img_red2 = filter_red2(img)
    thresh = combine(img_red1, img_red2)

    lines = line_detect(img, thresh)
    img_check = np.zeros(img.shape, np.uint8)   #generating black image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2-y1)/(x2-x1)
            if ((slope <= -0.5)&(slope>=-1)) | ((slope >= 0.5)&(slope<=1)):
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)   #marking red line to red
                cv2.line(img_check, (x1, y1), (x2, y2), (255, 255, 255), 2) #marking white line on black image
            else:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)   #marking not red line to green
    return img, img_check


if __name__ == '__main__':
    # video mode
    cap = cv2.VideoCapture("/media/anson/Disk1/demo/data/youbike/0827/5.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7000)
    while cv2.waitKey(1) < 1:

        (grabbed, frame) = cap.read()
        if not grabbed:
            exit()
        frame_result, frame_check = image(frame)
        cv2.imshow("result", frame_result)
        time.sleep(1/30)  # 可調速度