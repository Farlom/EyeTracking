import cv2
from mp_gaze_tracking import GazeTracking
import mediapipe as mp
import time
import numpy as np

image = cv2.VideoCapture('evaluation_images/night.jpg')
_, img = image.read()

# webcam = cv2.VideoCapture('60cm/510_.png')
webcam = cv2.VideoCapture('../../tools/output_2105_1.avi')
# webcam = cv2.VideoCapture('../videos/output_120cm.avi') # pupil detection evaluation
# webcam = cv2.VideoCapture('../videos/crop.mov')
# webcam = cv2.VideoCapture(1)
# webcam = cv2.VideoCapture('../photos/550_center.png')
# webcam = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.234:554") # cam 1
# webcam = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.235:554") # cam 2
# webcam.set(cv2.CAP_PROP_FPS, 1)
# webcam = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.236:554") # cam 3

SOURCE_FPS = webcam.get(cv2.CAP_PROP_FPS)

# fps
prev_frame_time = 0
new_frame_time = 0


roi = np.load('roi.npy')
x, y, w, h = roi

# frame_width = int(webcam.get(3))
frame_width = w

# frame_height = int(webcam.get(4))
frame_height = h

# out = cv2.VideoWriter('outpy_1005.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
counter = 0
calibration_counter = 0
lalala = 0

newcameramtx = np.load('newcameramtx.npy')
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

with mp.solutions.face_detection.FaceDetection(model_selection=1) as detector:
    gaze = GazeTracking(detector, (frame_width, frame_height))

    while True:
        # We get a new frame from the webcam
        ret, frame = webcam.read()
        boo = False

        if not ret:
            break

        if lalala == SOURCE_FPS or boo:
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



            # out.write(frame)
            counter += 1
            # cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # cv2.imwrite('word/frame_sectors.png', frame)

            frame = cv2.resize(frame, (3060, 1600))
            cv2.imshow("Demo", img)

            # frame = cv2.resize(frame, (3840, 2160))
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
            #                       cv2.WINDOW_FULLSCREEN)
            # cv2.imshow(window_name, image)
            # cv2.imwrite('60cm_2.png', frame)
            if cv2.waitKey(1) == 27:
                break
        lalala += 1


        # while True:
        #     ...

    webcam.release()
    # out.release()
    image.release()
    cv2.destroyAllWindows()

