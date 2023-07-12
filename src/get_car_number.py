import cv2
import time
import os
import numpy as np

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("model/numbers.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet(
    "model/yolov4-tiny-numbers.weights", "model/yolov4-tiny-numbers.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def detect_car_number(frame):
    detect_list = []
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, (0, 0, 255), 1)
        detect_list.append([class_names[classid], box[0]])

    detect_list.sort(key=lambda x: x[1])
    detect_list = np.asarray(detect_list)
    if len(detect_list) == 0:
        car_num = ''
    else:
        car_num = ''.join(str(e) for e in detect_list[:, 0])
    print('car number = ', car_num)
    cv2.imshow("output", frame)
    return car_num
