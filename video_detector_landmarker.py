import numpy as np
import cv2
import mediapipe as mp

LEFT_IRIS = [469, 470, 471, 470]  # x1 y1 x2 y2

cap = cv2.VideoCapture('IMG_0977.mp4')
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
 ret, frame = cap.read()
 # frame = cv2.flip(frame, 1)

 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 output = face_mesh.process(rgb_frame)
 landmark_points = output.multi_face_landmarks
 frame_h, frame_w, _ = frame.shape

 if landmark_points:
     mesh_points = np.array([np.multiply([p.x, p.y], [1920, 1080]).astype(int) for p in landmark_points[0].landmark])
     (cX, cY), rad = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
     landmarks = landmark_points[0].landmark
     center = np.array([cX, cY], dtype=np.int32)
     cv2.circle(frame, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)
 if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 cv2.imshow('frame', gray)
 if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()