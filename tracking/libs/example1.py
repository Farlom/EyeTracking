"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from mp_gaze_tracking import GazeTracking
import mediapipe as mp
import time
from PIL import Image, ImageOps
import numpy as np
import settings
from koeffs import BarrelDeformer

# gaze = GazeTracking()
# webcam = cv2.VideoCapture('60cm/510_.png')
# webcam = cv2.VideoCapture('../../tools/output_1605_1.avi')
webcam = cv2.VideoCapture('../videos/output_120cm.avi')
# webcam = cv2.VideoCapture('../videos/crop.mov')
# webcam = cv2.VideoCapture(1)
# webcam = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.234:554") # cam 1
# webcam = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.235:554") # cam 2
# webcam.set(cv2.CAP_PROP_FPS, 1)
# webcam = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.236:554") # cam 3

prev_frame_time = 0
new_frame_time = 0
arr = []
roi = np.load('roi.npy')
x, y, w, h = roi

# frame_width = int(webcam.get(3))
frame_width = w

# frame_height = int(webcam.get(4))
frame_height = h

out = cv2.VideoWriter('outpy_1005.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
counter = 0
calibration_counter = 0
x1 = None
x2 = None
y1 = None
y2 = None
angle = None
lalala = 0
newcameramtx = np.load('newcameramtx.npy')
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

with mp.solutions.face_detection.FaceDetection(model_selection=1) as detector:
    gaze = GazeTracking(detector, (frame_width, frame_height))

    while True:
        # We get a new frame from the webcam
        ret, frame = webcam.read()
        boo = True
        if lalala == 10 or boo:
            lalala = 0
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            frame = frame[y:y + h, x:x + w]
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)
            frame_copy = frame.copy()
            frame = gaze.annotated_frame()
            text = ""

            # if counter % 10 == 0:
            #     cv2.imwrite(f'./150cm_cal/{counter}_{text}.png', frame_copy)
            # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            # left_pupil = gaze.pupil_left_coords()
            # right_pupil = gaze.pupil_right_coords()
            # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.circle(frame, left_pupil, 1, (255, 255,255), 2, cv2.LINE_AA)
            new_frame_time = time.time()

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            fps = str(int(fps))
            cv2.putText(frame, f'FPS: {fps}', (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
            cv2.putText(frame, f'FPS: {fps}', (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            if calibration_counter == 0:
                # cv2.circle(frame, (0,0), 50, (0, 0, 255), -1)
                ...
            elif calibration_counter ==1:
                cv2.circle(frame, (frame_width, 0), 50, (0, 0, 255), -1)
            elif calibration_counter == 2:
                cv2.circle(frame, (frame_width, frame_height), 50, (0, 0, 255), -1)
            elif calibration_counter == 3:
                cv2.circle(frame, (0,frame_height), 50, (0, 0, 255), -1)
            else:
                for i in range(4):
                    cv2.circle(frame, (arr[i]), 30, (255, 255, 255), -1)
            if cv2.waitKey(1) & 0xff == ord('s'):
                if calibration_counter <= 3:
                    calibration_counter += 1
                    arr.append(gaze.vector)
                else:
                    print(arr)

            out.write(frame)
            counter += 1
            cv2.namedWindow("Demo", cv2.WINDOW_FULLSCREEN)

            frame = cv2.resize(frame, (2560, 1600))
            # frame = cv2.resize(frame, (3840, 2160))

            cv2.imshow("Demo", frame)
            # cv2.imwrite('60cm_2.png', frame)
            if cv2.waitKey(1) == 27:
                break
        lalala += 1


        # while True:
        #     ...

    webcam.release()
    out.release()

    cv2.destroyAllWindows()

