"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from mp_gaze_tracking import GazeTracking
import mediapipe as mp
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# gaze = GazeTracking()
webcam = cv2.VideoCapture('../videos/output_exp1.avi')
# webcam = cv2.VideoCapture(1)
# webcam = cv2.VideoCapture('../../../../photos/img.png')

prev_frame_time = 0
new_frame_time = 0

frame_width = int(webcam.get(3))
frame_height = int(webcam.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
counter = 0

with mp.solutions.face_detection.FaceDetection(model_selection=1) as detector:
    gaze = GazeTracking(detector)

    while True:
        # We get a new frame from the webcam
        ret, frame = webcam.read()

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

