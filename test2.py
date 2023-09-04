import cv2
import mediapipe as mp
import numpy as np

LEFT_IRIS = [469, 470, 471, 470]  # x1 y1 x2 y2

screen_w = 1920
screen_h = 1080
cropped_frame_size = 500
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

counter = 0
i_x = [0] * 4
i_y = [0] * 4
dw = 0
dh = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1600, 900), interpolation=cv2.INTER_AREA)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        mesh_points = np.array([np.multiply([p.x, p.y], [1600, 900]).astype(int) for p in landmark_points[0].landmark])
        (cX, cY), rad = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        landmarks = landmark_points[0].landmark
        center = np.array([cX, cY], dtype=np.int32)
        cv2.circle(frame, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)

        # for id, landmark in enumerate(landmarks[474:478]):
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #
        #     cv2.circle(frame, (x, y), 3, (0, 255, 0))
        #     if id == 1:
        #         screen_x = screen_w * landmark.x
        #         screen_y = screen_h * landmark.y
        #         # pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        # for landmark in left:
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #     cv2.circle(frame, (x, y), 3, (0, 255, 255))
        # if (left[0].y - left[1].y) < 0.004:
        #     print('blink', left[0].x * screen_w)
        # print(int(landmark_points[0].landmark[30].x * 1600), int(landmark_points[0].landmark[30].y * 900))

        # # top
        # cv2.circle(frame, (int(landmark_points[0].landmark[10].x * 1600), int(landmark_points[0].landmark[10].y * 900)), 5, (255, 0, 255))
        # # left
        # cv2.circle(frame, (int(landmark_points[0].landmark[123].x * 1600), int(landmark_points[0].landmark[123].y * 900)), 5, (255, 0, 255))
        # # bottom
        # cv2.circle(frame, (int(landmark_points[0].landmark[200].x * 1600), int(landmark_points[0].landmark[200].y * 900)), 5, (255, 0, 255))
        # # right
        # cv2.circle(frame, (int(landmark_points[0].landmark[352].x * 1600), int(landmark_points[0].landmark[352].y * 900)), 5, (255, 0, 255))

        x = int(landmark_points[0].landmark[123].x * 1600)
        y = int(landmark_points[0].landmark[10].y * 900)
        w = int(landmark_points[0].landmark[352].x * 1600)
        h = int(landmark_points[0].landmark[200].y * 900)
        # print(x, y, w, h)
        cropped_frame = frame[y:h, x:w]

        cropped_frame = cv2.resize(cropped_frame, (cropped_frame_size, cropped_frame_size), interpolation=cv2.INTER_AREA)

        rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        output_cropped = face_mesh.process(rgb_cropped_frame)
        landmark_points_cropped = output_cropped.multi_face_landmarks
        cropped_frame_h, cropped_frame_w, _ = frame.shape

        # if landmark_points_cropped:
        #     mesh_points_cropped = np.array([np.multiply([p.x, p.y], [cropped_frame_size, cropped_frame_size]).astype(int) for p in landmark_points_cropped[0].landmark])
        #     (cX, cY), rad = cv2.minEnclosingCircle(mesh_points_cropped[LEFT_IRIS])
        #     landmarks_cropped = landmark_points_cropped[0].landmark
        #     center = np.array([cX, cY], dtype=np.int32)
        #     cv2.circle(cropped_frame, center, int(rad), (0, 255, 0), 1, cv2.LINE_AA)
        #     # cv2.circle(cropped_frame, (int(landmark_points_cropped[0].landmark[469].x * (w-x)), int(landmark_points_cropped[0].landmark[469].y * (h-y))), 5, (255, 0, 255))
        #     # cv2.circle(cropped_frame, (int(landmark_points_cropped[0].landmark[470].x * (w - x)),
        #     #                            int(landmark_points_cropped[0].landmark[470].y * (h - y))), 5, (255, 0, 255))
        #     # cv2.circle(cropped_frame, (int(landmark_points_cropped[0].landmark[471].x * (w - x)),
        #     #                            int(landmark_points_cropped[0].landmark[471].y * (h - y))), 5, (255, 0, 255))

        cv2.imshow('crop', cropped_frame)
        # print(ret)

        # cv2.circle(frame, (x, y), 3, (0, 255, 255))
        # калибровка
        if cv2.waitKey(1) & 0xFF == ord('s'):
            i_x[counter] = int(center[0])
            i_y[counter] = int(center[1])
            counter += 1
            print(f'x:{center[0]}  y:{center[1]}')
            print(f'cropped x:{x}  cropped y:{y}')
            print(f'normalized x: {int((center[0] - x))} normalized y: {int((center[1] - y))}')
            # print(landmark_points[0].landmark[30].x * 1600)


        if counter == 0:
            cv2.circle(frame, (0, 0), 50, (0, 0, 255), -1)


        if counter == 1:
            cv2.circle(frame, (1600, 0), 50, (0, 0, 255), -1)
            x2, y2 = center

        if counter == 2:
            cv2.circle(frame, (0, 900), 50, (0, 0, 255), -1)
            x3, y3 = center

        if counter == 3:
            cv2.circle(frame, (1600, 900), 50, (0, 0, 255), -1)
            x4, y4 = center

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
            # cv2.circle(frame, (int((center[0] - x) / 500 * 1600), 100), 50, (0, 0, 255), -1)
            cv2.circle(frame, (
                (center[0] - i_x[0]) * dw,
                (center[1] - i_y[0]) * dw), 50, (0, 0, 255), -1)
            # print(center[0] - (i_x[0] / 500 * 1600))

            # print(center[0] / 500 * dw)
            # print(int((center[0] - i_x[0]) * (cropped_frame_size / dw)))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
