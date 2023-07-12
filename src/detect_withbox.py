import cv2
import time
import line_detect as line_detect
import numpy as np
import both_in_one as license_detect


def matchCheck(video_dir, save_folder, start_frame):
    CONFIDENCE_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    class_names = []
    with open("model/coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # start from frame 6000th

    net = cv2.dnn.readNet("model/yolov4-tiny-car.weights",
                          "model/yolov4-tiny-car.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    i = 0
    while cv2.waitKey(1) < 1:
        start = time.time()
        flag_save = False
        (grabbed, frame) = cap.read()
        if not grabbed:
            exit()
        box_check = np.zeros(frame.shape, np.uint8)  # generate box check

        classes, scores, boxes = model.detect(
            frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        frame, line_check = line_detect.image(
            frame)    # call line detect function

        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            if (classid == 2) | (classid == 7):     # choose only car and truck
                label = "%s : %f" % (class_names[classid], score)
                print(box)
                cv2.rectangle(
                    frame, [box[0], box[1], box[2], box[3]+15], color, 2)
                # fill up box_check
                cv2.rectangle(
                    box_check, [box[0], box[1], box[2], box[3]+15], (255, 255, 255), -1)
                cv2.putText(frame, label, (box[0], box[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # match line and box
        img_capture = line_check & box_check
        cv2.imshow("img_check", img_capture)

        # check whether line and box match
        img_dark = np.zeros(img_capture.shape, np.uint8)
        difference = cv2.subtract(img_capture, img_dark)

        if np.any(difference):
            frame = license_detect.get_car_num(frame)
            flag_save = True

        else:
            print(".")

        end = time.time()
        fps = "FPS: %.2f " % (1 / (end - start))
        cv2.putText(frame, fps, (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if flag_save:

            cv2.imwrite(save_folder+'/save'+str(i)+'.png', frame)
            print("saved")

        cv2.imshow("output", frame)

        i += 1
        if (end-start) < (1/30):
            time.sleep(1/30-(end-start))


if __name__ == '__main__':
    video_dir = 'media/1080.webm'
    save_folder = 'media/img_test'
    matchCheck(video_dir, save_folder, start_frame=350)
