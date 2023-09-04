import cv2
import mediapipe as mp
import numpy as np
import matplotlib as plt

cap = cv2.VideoCapture(0)

model_path = 'face_landmarker.task'

ndarray = np.full((900, 1600, 3), 0, dtype=np.uint8)

# show image


LEFT_IRIS = [469, 470, 471, 470]  # x1 y1 x2 y2
LEFT_EYE2 = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 367, 386, 385, 384, 398]
global_x = 100
global_y = 100

top_eye_x = 0
top_eye_y = 0
bottom_eye_x = 0
bottom_eye_y = 0

LEFT_EYE = [130, 223, 243, 230]  # x y w h

counter = 0  # !!!!!!!
configuration = [0] * 4
ratio_x = 0
ratiox_y = 0

cX = 0
cY = 0
rad = 0


mesh_points = 0
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global mesh_points
    mesh_points = np.array([np.multiply([p.x, p.y], [640, 480]).astype(int) for p in result.face_landmarks[0]])
    global cX, cY, rad
    (cX, cY), rad = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])

    global top_eye_y, top_eye_x, bottom_eye_x, bottom_eye_y
    top_eye_x = int(result.face_landmarks[0][130].x * 640)
    top_eye_y = int(result.face_landmarks[0][223].y * 480)
    bottom_eye_x = int(result.face_landmarks[0][243].x * 640)
    bottom_eye_y = int(result.face_landmarks[0][230].y * 480)
    # print(top_eye_x, top_eye_y)
    # top_eye_x = result.face_landmarks[0][257].x
    # top_eye_y = result.face_landmarks[0][257].y

    # cv2.imwrite('test.jpg', output_image.numpy_view())
    # cv2.imshow('frame', np.full((100, 100, 3), 0, dtype=np.uint8))


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        cap_height, cap_width = frame.shape[:2]
        # print(cap_width, cap_height)
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        #
        # cv2.circle(frame, landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))), 2, (255, 0, 0))
        landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        # landmarker.
        # print(coords)
        # cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.circle(frame, (global_x,global_y), 5, (0,255,0))
        # FaceLandmarkerResult(landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))))
        center = np.array([cX, cY], dtype=np.int32)
        cv2.circle(frame, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.circle(frame, (top_eye_x, top_eye_y), 10, (0, 255, 0), 1, cv2.LINE_AA)
        # print(mesh_points[LEFT_EYE])
        cv2.rectangle(frame, (top_eye_x, top_eye_y), (bottom_eye_x, bottom_eye_y), (0, 255, 0), 1)
        # print(top_eye_x, bottom_eye_x, top_eye_y, bottom_eye_y)
        print(center[0], center[1])
        # print(top_eye_x, bottom_eye_x, top_eye_y, bottom_eye_y)
        # new_frame = frame[top_eye_x:bottom_eye_x, top_eye_y:bottom_eye_y]
        # new_frame = frame[cX-50:cX+50, cY-50:cY+50]
        # new_frame = frame[20:100, 20:100]
        # new_frame = cv2.resize(new_frame, (500, 500))
        # print(new_frame)
        # cv2.imshow('new', new_frame)
        # cv2.polylines(frame, [mesh_points[LEFT_EYE2]], True, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.circle(ndarray, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.circle(frame, (top_eye_x, top_eye_y), 5,(0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imshow('canvas', ndarray)
        # ndarray = np.full((900, 1600, 3), 0, dtype=np.uint8)

        cv2.imshow('img', frame)

        # print(frame[1][0], 1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            landmarker.close()
            break

        elif key == ord('z'):
            if counter == 0:
                configuration[counter] = np.array([cX, cY], dtype=np.int32)
                print(configuration[counter][0])
                counter += 1
            elif counter == 1:
                configuration[counter] = np.array([cX, cY], dtype=np.int32)
                print(configuration[counter])
                counter += 1
            elif counter == 2:
                configuration[counter] = np.array([cX, cY], dtype=np.int32)
                print(configuration[counter])
                counter += 1
            elif counter == 3:
                configuration[counter] = np.array([cX, cY], dtype=np.int32)
                print(configuration[counter])
                counter += 1
                ratio_x =  (configuration[1] - configuration[0]) / 1600
            # print('123')
            elif counter == 4:
                cv2.circle(ndarray, center, 20, (0, 255, 0), 1, cv2.LINE_AA)
        if counter == 4:
            print((configuration[0] - center[0]) * ratio_x)
            # cv2.circle(frame, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)
cap.release()
cv2.destroyAllWindows()