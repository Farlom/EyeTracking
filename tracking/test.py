from l2cs import Pipeline, render
import cv2
import pathlib
import torch
import mediapipe as mp
import numpy as np
import math

CWD = pathlib.Path.cwd()

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')  # or 'gpu'
)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    with mp.solutions.face_detection.FaceDetection() as detector:
        results = detector.process(frame)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            x = round(bbox.xmin * width)
            y = round(bbox.ymin * height)
            width = round(bbox.width * width)
            height = round(bbox.height * height)
            # print(results.detections[0].location_data.relative_bounding_box)
            cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 255, 255))
            face = frame[y:y+height, x:x+width]
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            res_face = gaze_pipeline.step(face)
            face = render(face, res_face)
            # print(res_face.pitch / np.pi * 180)
            # print(res_face.yaw / np.pi * 180)
            # print(res_face.)

            DISTANCE_TO_OBJECT = 1000  # mm
            HEIGHT_OF_HUMAN_FACE = 250  # mm

            image_height, image_width = frame.shape[:2]
            # print(results.bboxes[0])
            # cv2.line(frame, (50, 50), (50, int(50+res_face.bboxes[0][3] - res_face.bboxes[0][1])), (255,255,255), 3)
            length_per_pixel = HEIGHT_OF_HUMAN_FACE / (res_face.bboxes[0][3] - res_face.bboxes[0][1])

            dx = -DISTANCE_TO_OBJECT * np.tan(res_face.pitch) / length_per_pixel
            # 100000000 is used to denote out of bounds
            dx = dx if not np.isnan(dx) else 100000000

            dy = -DISTANCE_TO_OBJECT * np.arccos(res_face.pitch) * np.tan(res_face.yaw) / length_per_pixel
            dy = dy if not np.isnan(dy) else 100000000
            gaze_point = int(image_width / 2 + dx), int(image_height / 2 + dy)

            cv2.circle(frame, gaze_point, 25, (0, 0, 255), -1)
            quadrants = [
                ("center",
                 (int(image_width / 4), int(image_height / 4), int(image_width / 4 * 3), int(image_height / 4 * 3))),
                ("top_left", (0, 0, int(image_width / 2), int(image_height / 2))),
                ("top_right", (int(image_width / 2), 0, image_width, int(image_height / 2))),
                ("bottom_left", (0, int(image_height / 2), int(image_width / 2), image_height)),
                ("bottom_right", (int(image_width / 2), int(image_height / 2), image_width, image_height)),
            ]

            for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
                if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                    # show in top left of screen
                    cv2.putText(frame, quadrant, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                    break

            cv2.line(frame, (0, 100), (DISTANCE_TO_OBJECT, 100), (255, 0,0), 3)
            cv2.line(frame, (0, 100), (DISTANCE_TO_OBJECT, 100 + round(100 * math.sin(math.radians(res_face.yaw[0] / np.pi * 180)))), (255, 0,0), 3)
            tmp = round(DISTANCE_TO_OBJECT * math.sin(math.radians(res_face.yaw[0] / np.pi * 180)))
            print(tmp, frame.shape)
            cv2.circle(frame, (frame.shape[1] // 2 , frame.shape[0] // 2 - tmp), 15, (0, 255, 0), -1)
            # print(math.sin(math.radians(res_face.yaw[0] / np.pi * 180)))
            # print(res_face.yaw[0])

            # if results:
            #     x_min = round(results.bboxes[0][0])
            #     y_min = round(results.bboxes[0][1])
            #     x_max = round(results.bboxes[0][2])
            #     y_max = round(results.bboxes[0][3])
            #
            #     bbox_width = x_max - x_min
            #     bbox_height = y_max - y_min
            #
            #     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            #     draw_gaze(x_min, y_min, bbox_width, bbox_height,frame, (results.pitch[0], results.yaw[0]), color=(0,0,255))

            cv2.imshow('face', face)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
# Process frame and visualize
# results = gaze_pipeline.step(frame)
# frame = render(frame, results)