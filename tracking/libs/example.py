import mediapipe as mp
import cv2
import numpy as np

LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# mediapipe pts
LEFT_EYE_IRIS = [469, 470, 471, 472]
RIGHT_EYE_IRIS = [474, 475, 476, 477]

cap = cv2.VideoCapture('face_frame.png')

with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as detector:
    while True:
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.process(frame).multi_face_landmarks

        if results:
            height, width = frame.shape[:2]
            landmarks = results[0].landmark
            landmarks = np.array(
                [np.multiply([p.x, p.y], [width, height]).astype(int) for p in
                 landmarks]
            )

            (cX_l, cY_l), rad_l = cv2.minEnclosingCircle(landmarks[LEFT_EYE_IRIS])
            (cX_r, cY_r), rad_r = cv2.minEnclosingCircle(landmarks[RIGHT_EYE_IRIS])

            center_l = np.array([cX_l, cY_l], dtype=np.int32)
            cv2.circle(frame, center_l, int(rad_l), (0, 255, 0), 1, cv2.LINE_AA)

            center_r = np.array([cX_r, cY_r], dtype=np.int32)
            cv2.circle(frame, center_r, int(rad_r), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.imwrite('word/mp_eye.png', frame)
        if cv2.waitKey(1) == 27:

            break
    cap.release()
    # out.release()

    cv2.destroyAllWindows()