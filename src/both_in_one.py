from operator import ge
from webbrowser import get
import cv2
import time
import os
from get_car_number import detect_car_number

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("model/license.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet(
    "model/yolov4-tiny-license-plate.weights", "model/yolov4-tiny-license-plate.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def get_car_num(frame):
    file_path = os.path.dirname(os.path.abspath(__file__))
    image_path = file_path + '/picture'
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    start = time.time()
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()
    car_num = None
    for (classid, score, box) in zip(classes, scores, boxes):
        x, y, w, h = box
        ymax, xmax = frame[:, :, 0].shape
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        if x+w < xmax and y+h < ymax:
            car_num = detect_car_number(
                frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
            cv2.putText(
                frame, car_num, (box[0], box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    fps = "FPS: %.2f " % (1 / (end - start))
    return frame


if __name__ == '__main__':

    cap = cv2.VideoCapture("media/1080.webm")
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 7000)  # set start frame
    while cv2.waitKey(1) < 1:
        (grabbed, frame) = cap.read()
        if not grabbed:
            exit()
        output = get_car_num(frame)
        cv2.imshow("output", output)
        time.sleep(1/30)
