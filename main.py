import cv2
import mediapipe as mp
import numpy as np
import matplotlib as plt

cap = cv2.VideoCapture(0)

model_path = 'face_landmarker.task'
LEFT_EYE = [469, 470, 471, 470]  # x1 y1 x2 y2
global_x = 100
global_y = 100

cX = 0
cY = 0
rad = 0

x1 = 0
x2 = 0
y1 = 0
y2 = 0

mesh_points = 0
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('face landmarker result: {}'.format(result.face_blendshapes))
    # print(result.face_landmarks[0][0].x, result.face_landmarks[0][0].y)
    # cv2.imshow('vid', frame)
    global global_x, global_y

    global mesh_points, x1, x2, y1, y2
    mesh_points = np.array([np.multiply([p.x, p.y], [640,480]).astype(int) for p in result.face_landmarks[0]])
    # output_image = output_image.numpy_view()
    # cv2.imshow('img', output_image.numpy_view())
    # print(mesh_points[469])
    global cX, cY, rad
    (cX, cY), rad = cv2.minEnclosingCircle(mesh_points[LEFT_EYE])
    # print(output_image.numpy_view(), 1)
    # cv2.imshow('frame', output_image.numpy_view())
    # print(cv2.minEnclosingCircle(mesh_points[LEFT_EYE]))
    # x1 = [int(result.face_landmarks[0][469].x * 640),
    # x2 = int(result.face_landmarks[0][469].x * 640)
    # y1 = int(result.face_landmarks[0][469].x * 640)
    # y2 = int(result.face_landmarks[0][469].x * 640)

    # global_x = int(result.face_landmarks[0][469].x * 640 + (((result.face_landmarks[0][469].x - result.face_landmarks[0][471].x) * 640) / 2))
    # global_y = int(((result.face_landmarks[0][470].y - result.face_landmarks[0][472].y) * 640) / 2)
    # # global_x = int(result.face_landmarks[0][469].x * 640)
    # global_y = int(result.face_landmarks[0][469].y * 480)

    # return result.face_landmarks[0][0].x * 640, result.face_landmarks[0][0].y * 480

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


        # cv2.circle(frame, landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))), 2, (255, 0, 0))
        landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        # landmarker.
        # print(coords)
        # cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.circle(frame, (global_x,global_y), 5, (0,255,0))
        # FaceLandmarkerResult(landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))))
        center = np.array([cX, cY], dtype=np.int32)
        cv2.circle(frame, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('img', frame)
        # print(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()