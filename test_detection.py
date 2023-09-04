import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FRAME_SIZE = 500

LEFT_IRIS = [469, 470, 471, 470]  # x1 y1 x2 y2

BACKGROUND = np.zeros((900,1600,3), np.uint8)

base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

counter = 0
i_x = [0] * 4
i_y = [0] * 4
dw = 0
dh = 0

while True:
    BACKGROUND = np.zeros((900, 1600, 3), np.uint8)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    for detection in detection_result.detections:
        bbox = detection.bounding_box

        # start_point = bbox.origin_x, bbox.origin_y
        # end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        # cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 3)

        face = frame[bbox.origin_y:bbox.origin_y + bbox.height, bbox.origin_x:bbox.origin_x + bbox.width]
        face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        # cv2.resize(frame, (1600, 900), interpolation=cv2.INTER_AREA)

        if landmark_points:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [FRAME_SIZE, FRAME_SIZE]).astype(int) for p in landmark_points[0].landmark])
            (cX, cY), rad = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            landmarks = landmark_points[0].landmark
            center = np.array([cX, cY], dtype=np.int32)

            cv2.circle(face, (int(landmarks[33].x * FRAME_SIZE), int(landmarks[33].y * FRAME_SIZE)), 1, (0, 0, 255))
            cv2.circle(face, (int(landmarks[133].x * FRAME_SIZE), int(landmarks[133].y * FRAME_SIZE)), 1, (0, 0, 255))

            cv2.circle(face, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)

            # print(center[0] - (int(landmarks[33].x * FRAME_SIZE)))

        if counter == 0:
            cv2.circle(BACKGROUND, (0, 0), 50, (0, 0, 255), -1)

        if counter == 1:
            cv2.circle(BACKGROUND, (1600, 0), 50, (0, 0, 255), -1)

        if counter == 2:
            cv2.circle(BACKGROUND, (0, 900), 50, (0, 0, 255), -1)

        if counter == 3:
            cv2.circle(BACKGROUND, (1600, 900), 50, (0, 0, 255), -1)

        if counter == 4:

            print(f'x: {i_x}')
            print(f'y: {i_y}')
            # print(f'{i_x[1] - i_x[0]}')
            # print(f'{i_x[3] - i_x[2]}')
            # print(f'y {i_y[1] - i_y[0]}')
            # print(f'{i_y[3] - i_y[2]}')
            # dw = round(abs(((i_x[1] - i_x[0]) + (i_x[3] - i_x[2])) / 2))
            dw = round(abs(((i_x[0] + i_x[2]) / 2) - ((i_x[1] + i_x[3]) / 2)))
            # dh = abs(((i_y[1] - i_y[0]) + (i_y[3] - i_y[2])) / 2)
            dh = round(abs(((i_y[0] + i_y[2]) / 2) - ((i_y[1] + i_y[3]) / 2)))
            print(dw, dh)
            counter += 1

        if counter == 5:
            cv2.circle(BACKGROUND, (
                (center[0] - i_x[0]) * dw,
                (center[1] - i_y[0]) * dw), 50, (0, 0, 255), -1)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            i_x[counter] = int(center[0] - landmarks[33].x * FRAME_SIZE)
            i_y[counter] = int(center[1])
            counter += 1
            # print(f'x:{center[0]}  y:{center[1]}')
            # print(f'cropped x:{x}  cropped y:{y}')
            # print(f'normalized x: {int((center[0] - x))} normalized y: {int((center[1] - y))}')
    cv2.imshow('background', BACKGROUND)
    cv2.imshow('face', face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break