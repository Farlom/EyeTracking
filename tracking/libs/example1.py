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
webcam = cv2.VideoCapture('../videos/output_exp2.avi')
# webcam = cv2.VideoCapture(1)
# webcam = cv2.VideoCapture('../photos/img.png')

prev_frame_time = 0
new_frame_time = 0
arr = []
frame_width = int(webcam.get(3))
frame_height = int(webcam.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
counter = 0
calibration_counter = 0
x1 = None
x2 = None
y1 = None
y2 = None
angle = None
with mp.solutions.face_detection.FaceDetection(model_selection=1) as detector:
    gaze = GazeTracking(detector)

    while True:
        # We get a new frame from the webcam
        ret, frame = webcam.read()
        # frame = cv2.flip(frame, 1)

        if settings.FIX_BARREL_DISTORTION:
            frame = Image.fromarray(frame)
            frame = ImageOps.deform(frame, BarrelDeformer(-0.21, 0, frame_width, frame_height))
            frame = np.asarray(frame)
        if not ret:
            break

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.circle(frame, left_pupil, 1, (255, 255,255), 2, cv2.LINE_AA)
        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = str(int(fps))
        cv2.putText(frame, f'FPS: {fps}', (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

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

        # out.write(frame)
        counter += 1
        cv2.imshow("Demo", frame)
        if cv2.waitKey(1) == 27:
            break


        # while True:
        #     ...

    webcam.release()
    out.release()

    cv2.destroyAllWindows()

