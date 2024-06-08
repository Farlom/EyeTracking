import cv2
import mediapipe as mp
import numpy as np
import utils
import math

count = 0
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
faces = []

cap = cv2.VideoCapture('./videos/output_exp1.avi')
# cap = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.234:554")
old_l_center_x = 0
old_l_center_y = 0
old_r_center_x = 0
old_r_center_y = 0

old_cX_l = 0
old_cY_l = 0
old_rad_l = 0
old_cX_r = 0
old_cY_r = 0
old_rad_r = 0

old_left_eye_mesh = np.array([[0, 0] for p in LEFT_EYE], dtype=np.int32)
old_right_eye_mesh = np.array([[0, 0] for p in RIGHT_EYE], dtype=np.int32)

EPSILLON = 5


def get_landmarks(img: np.ndarray):
    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as mesh:
        return mesh.process(detected_face).multi_face_landmarks


def get_faces(img: np.ndarray, detector: mp.solutions.face_detection.FaceDetection):
    results = detector.process(img)
    bbox = None
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        # print(results.detections[0].location_data.relative_bounding_box)
        cv2.rectangle(img, (int(bbox.xmin * 1920), int(bbox.ymin * 1080)),
                      (int((bbox.xmin + bbox.width) * 1920), int((bbox.ymin + bbox.height) * 1080)), (255, 0, 0), -1)
        # recursive = get_faces(img, detector)
        faces.append(bbox)
        return bbox, get_faces(img, detector)

    else:
        return


with mp_pose.FaceDetection(model_selection=1) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            get_faces(image, detector)
        except:
            print('Recursion err')
        if faces:
            if len(faces) == 1:
                face = faces[0]
            else:
                dist = 1920
                face = faces[0]
                for i in range(len(faces)):
                    cv2.rectangle(frame,
                                  (int(faces[i].xmin * 1920), int(faces[i].ymin * 1080)),
                                  (int((faces[i].xmin + faces[i].width) * 1920),
                                   int((faces[i].ymin + faces[i].height) * 1080)),
                                  (255, 0, 0),
                                  1)
                    if ((faces[i].xmin + faces[i].width) / 2) * 1920 < dist:
                        dist = ((faces[i].xmin + faces[i].width) / 2) * 1920
                        face = faces[i]

            x = int(face.xmin * 1920)
            y = int(face.ymin * 1080)
            w = int((face.xmin + face.width) * 1920)
            h = int((face.ymin + face.height) * 1080)

            detected_face = frame[y:h, x:w]

            cv2.rectangle(frame,
                          (int(face.xmin * 1920), int(face.ymin * 1080)),
                          (int((face.xmin + face.width) * 1920), int((face.ymin + face.height) * 1080)),
                          (255, 255, 0),
                          1)

            landmark_points = get_landmarks(detected_face)
            # with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            #     landmark_points = face_mesh.process(detected_face).multi_face_landmarks
            if landmark_points:
                mesh_points = np.array(
                    [np.multiply([p.x, p.y], [w - x, h - y]).astype(int) for p in landmark_points[0].landmark]
                )

                (cX_l, cY_l), rad_l = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (cX_r, cY_r), rad_r = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                if abs(cX_l - old_cX_l) >= EPSILLON or abs(cY_l - old_cY_l) >= EPSILLON:
                    old_cX_l = cX_l
                    old_cY_l = cY_l
                    old_rad_l = rad_l
                else:
                    cX_l = old_cX_l
                    cY_l = old_cY_l
                    rad_l = old_rad_l

                if abs(cX_r - old_cX_r) >= EPSILLON or abs(cY_r - old_cY_r) >= EPSILLON:
                    old_cX_r = cX_r
                    old_cY_r = cY_r
                    old_rad_r = rad_r
                else:
                    cX_r = old_cX_r
                    cY_r = old_cY_r
                    rad_r = old_rad_r

                center_l = np.array([cX_l, cY_l], dtype=np.int32)
                cv2.circle(detected_face, center_l, int(rad_l), (255, 255, 255), 1, cv2.LINE_AA)

                center_r = np.array([cX_r, cY_r], dtype=np.int32)
                cv2.circle(detected_face, center_r, int(rad_r), (255, 255, 255), 1, cv2.LINE_AA)

                left_eye_mesh = np.array([mesh_points[p] for p in LEFT_EYE], dtype=np.int32)
                right_eye_mesh = np.array([mesh_points[p] for p in RIGHT_EYE], dtype=np.int32)

                for i in range(len(LEFT_EYE)):
                    if abs(left_eye_mesh[i][0] - old_left_eye_mesh[i][0]) >= EPSILLON or abs(
                            left_eye_mesh[i][1] - old_left_eye_mesh[i][1]) >= EPSILLON:
                        old_left_eye_mesh = left_eye_mesh
                    else:
                        left_eye_mesh = old_left_eye_mesh

                for i in range(len(RIGHT_EYE)):
                    if abs(right_eye_mesh[i][0] - old_right_eye_mesh[i][0]) >= EPSILLON or abs(
                            right_eye_mesh[i][1] - old_right_eye_mesh[i][1]) >= EPSILLON:
                        old_right_eye_mesh = right_eye_mesh
                    else:
                        right_eye_mesh = old_right_eye_mesh

                cv2.polylines(detected_face, [left_eye_mesh],
                              True,
                              (0, 255, 0),
                              1, cv2.LINE_AA)
                cv2.polylines(detected_face, [right_eye_mesh],
                              True,
                              (0, 255, 0),
                              1, cv2.LINE_AA)

                # nose x
                nose_x_axis_k = (mesh_points[278][1] - mesh_points[48][1]) / (
                        mesh_points[278][0] - mesh_points[48][0])
                nose_x_axis_b = mesh_points[48][1] - (nose_x_axis_k * mesh_points[48][0])
                nose_x_axis_y1 = int(nose_x_axis_b)
                nose_x_axis_y2 = int(detected_face.shape[1] * nose_x_axis_k + nose_x_axis_b)
                cv2.line(detected_face, (0, nose_x_axis_y1),
                         (detected_face.shape[1], nose_x_axis_y2), (0, 0, 255), 1, cv2.LINE_AA)

                # nose y axis
                if mesh_points[200][0] - mesh_points[151][0] != 0:
                    nose_y_axis_k = (mesh_points[200][1] - mesh_points[151][1]) / (
                            mesh_points[200][0] - mesh_points[151][0])
                    nose_y_axis_b = mesh_points[151][1] - (nose_y_axis_k * mesh_points[151][0])
                    nose_y_axis_y1 = int(nose_y_axis_b)
                    nose_y_axis_y2 = int(detected_face.shape[1] * nose_y_axis_k + nose_y_axis_b)
                    cv2.line(detected_face, (0, nose_y_axis_y1),
                             (detected_face.shape[1], nose_y_axis_y2), (0, 255, 0), 1, cv2.LINE_AA)
                    # cv2.line(detected_face, (mesh_points[151][0], mesh_points[151][1]),
                    #          (mesh_points[200][0], mesh_points[200][1]), (0, 255, 0), 1, cv2.LINE_AA)

                    # пересечение
                    a = np.array([[1, -nose_x_axis_k], [1, -nose_y_axis_k]])
                    b = np.array([nose_x_axis_b, nose_y_axis_b])
                    nose_x_axis_center = int(np.linalg.solve(a, b)[1])
                    nose_y_axis_center = int(np.linalg.solve(a, b)[0])
                    cv2.circle(detected_face, (nose_x_axis_center, nose_y_axis_center), 2, (255, 255, 255), 2,
                               cv2.LINE_AA)

                    # вектор носа
                    cv2.line(detected_face, (mesh_points[4][0], mesh_points[4][1]),
                             (nose_x_axis_center, nose_y_axis_center), (255, 0, 0), 1, cv2.LINE_AA)

                # nose
                cv2.circle(detected_face, (mesh_points[4][0], mesh_points[4][1]), 2, (255, 0, 0), 2,
                           cv2.LINE_AA)

                shift_x = mesh_points[4][0] - nose_x_axis_center
                shift_y = mesh_points[4][1] - nose_y_axis_center

                # cv2.line(detected_face, (mesh_points[48][0], mesh_points[48][1]),
                #          (mesh_points[278][0], mesh_points[278][1]), (0, 0, 255), 1, cv2.LINE_AA)
                nose_x_center = int(mesh_points[48][0] + (mesh_points[278][0] - mesh_points[48][0]) / 2)
                nose_x_center_y = int(mesh_points[48][1] + (mesh_points[278][1] - mesh_points[48][1]) / 2)

                left_eye_coords = [mesh_points[p] for p in LEFT_EYE]
                l_max_x = (max(left_eye_mesh, key=lambda item: item[0]))[0]
                l_min_x = (min(left_eye_mesh, key=lambda item: item[0]))[0]
                l_max_y = (max(left_eye_mesh, key=lambda item: item[1]))[1]
                l_min_y = (min(left_eye_mesh, key=lambda item: item[1]))[1]
                l_width = l_max_x - l_min_x
                l_height = l_max_y - l_min_y
                l_center_x = int(l_min_x + (l_width / 2))
                l_center_y = int(l_min_y + (l_height / 2))

                right_eye_coords = [mesh_points[p] for p in RIGHT_EYE]
                # r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
                # r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
                # r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
                # r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]
                r_max_x = (max(right_eye_mesh, key=lambda item: item[0]))[0]
                r_min_x = (min(right_eye_mesh, key=lambda item: item[0]))[0]
                r_max_y = (max(right_eye_mesh, key=lambda item: item[1]))[1]
                r_min_y = (min(right_eye_mesh, key=lambda item: item[1]))[1]
                r_width = r_max_x - r_min_x
                r_height = r_max_y - r_min_y
                r_center_x = int(r_min_x + (r_width / 2))
                r_center_y = int(r_min_y + (r_height / 2))

                left_ratio_h = (cX_l - l_min_x) / l_width
                left_ratio_v = (cY_l - l_min_y) / l_height
                right_ratio_h = (cX_r - r_min_x) / r_width
                right_ratio_v = (cY_r - r_min_y) / r_height

                cv2.line(detected_face, (r_center_x - shift_x, r_center_y - shift_y),
                         (int(cX_r), int(cY_r)), (255, 255, 0), 3, cv2.LINE_AA)
                cv2.line(detected_face, (l_center_x - shift_x, l_center_y - shift_y),
                         (int(cX_l), int(cY_l)), (255, 255, 0), 3, cv2.LINE_AA)

                r_center_x -= shift_x
                l_center_x -= shift_x
                r_center_y -= shift_y
                l_center_y -= shift_y

                # расчет уравнения прямой правого глаза
                if r_center_x != cX_r and r_center_y != cY_r:
                    tmp_k_r = (r_center_y - cY_r) / (r_center_x - cX_r)
                    tmp_b_r = cY_r - (tmp_k_r * cX_r)
                # расчет уравнения прямой левого глаза
                if l_center_x != cX_l and l_center_y != cY_l:
                    tmp_k_l = (l_center_y - cY_l) / (l_center_x - cX_l)
                    tmp_b_l = cY_l - (tmp_k_l * cX_l)

                if r_center_x > cX_r:
                    cv2.line(detected_face, (r_center_x, r_center_y),
                             (0, int(tmp_b_r)), (255, 0, 255), 1, cv2.LINE_AA)
                    if int(tmp_b_r) < detected_face.shape[0]:
                        if int(tmp_b_r) > 0:
                            sub_img = frame[0:0 + frame.shape[0], 0:300]
                            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                            frame[0:0 + frame.shape[0], 0:300] = res
                        else:
                            sub_img = frame[0:150, 0:frame.shape[1]]
                            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                            frame[0:150, 0:frame.shape[1]] = res
                    else:
                        sub_img = frame[frame.shape[0]-150:frame.shape[0], 0:frame.shape[1]]
                        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                        frame[frame.shape[0]-150:frame.shape[0], 0:frame.shape[1]] = res
                else:
                    cv2.line(detected_face, (r_center_x, r_center_y),
                             (detected_face.shape[1], int((detected_face.shape[1] * tmp_k_r) + tmp_b_r)), (255, 0, 255), 1, cv2.LINE_AA)
                    if int(tmp_b_r) < detected_face.shape[0]:
                        if int(tmp_b_r) > 0:
                            sub_img = frame[0:0 + frame.shape[0], frame.shape[1] - 300:frame.shape[1]]
                            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                            frame[0:0 + frame.shape[0], frame.shape[1] - 300:frame.shape[1]] = res
                    else:
                        sub_img = frame[0:150, 0:frame.shape[1]]
                        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                        frame[0:150, 0:frame.shape[1]] = res

                if l_center_x > cX_l:
                    cv2.line(detected_face, (l_center_x, l_center_y),
                             (0, int(tmp_b_l)), (255, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.line(detected_face, (l_center_x, l_center_y),
                             (detected_face.shape[1], int((detected_face.shape[1] * tmp_k_l) + tmp_b_l)),
                             (255, 0, 255), 1, cv2.LINE_AA)

                # if r_center_y > cY_r:
                #
                #     cv2.line(detected_face, (r_center_x, r_center_y),
                #              (int((0 - tmp_b_r) / tmp_k_r), int(0)), (255, 0, 255), 1, cv2.LINE_AA)
                #     cv2.line(detected_face, (r_center_x, r_center_y),
                #              (0, int(tmp_b_r)), (255, 255, 255), 2, cv2.LINE_AA)
                    # if int((0 - tmp_b_r) / tmp_k_r) <= 0:
                    #     sub_img = frame[0:0 + frame.shape[0], 0:300]
                    #     white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                    #     res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    #
                    #     frame[0:0 + frame.shape[0], 0:300] = res
                    # elif int((0 - tmp_b_r) / tmp_k_r) > 0:
                    #     sub_img = frame[0:0 + frame.shape[0], frame.shape[1] - 300:frame.shape[1]]
                    #     white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                    #     res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    #
                    #     frame[0:0 + frame.shape[0], frame.shape[1] - 300:frame.shape[1]] = res
                    # elif
                # else:
                #     cv2.line(detected_face, (r_center_x, r_center_y),
                #              (int((detected_face.shape[0] - tmp_b_r) / tmp_k_r), detected_face.shape[0]),
                #              (255, 0, 255), 1, cv2.LINE_AA)

                # if l_center_y > cY_l:
                #     cv2.line(detected_face, (l_center_x, l_center_y),
                #              (int((0 - tmp_b_l) / tmp_k_l), int(0)), (255, 0, 255), 1, cv2.LINE_AA)
                #     if int((0 - tmp_b_l) / tmp_k_l) <= 0:
                #         sub_img = frame[0:0 + frame.shape[0], 0:300]
                #         white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                #         res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                #
                #         frame[0:0 + frame.shape[0], 0:300] = res
                # else:
                #     cv2.line(detected_face, (l_center_x, l_center_y),
                #              (int((detected_face.shape[0] - tmp_b_l) / tmp_k_l), detected_face.shape[0]),
                #              (255, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('detected face', detected_face)
        cv2.imshow('Mediapipe Feed', frame)

        count += 1
        # if count % 10 == 0:
        #     cv2.imwrite(f'imgs/exp1/{count}.jpg', frame)
        faces = []
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# if landmark_points:
#     mesh_points = utils.get_mesh_points(detected_face, landmark_points)
#     (cX_l, cY_l), rad_l = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
#     (cX_r, cY_r), rad_r = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

# print(mesh_points[94][0])
# cv2.circle(detected_face,(mesh_points[94][0], mesh_points[94][1]), 2, (255, 255, 255), 1, cv2.LINE_AA)
# cv2.circle(detected_face, (mesh_points[1][0], mesh_points[1][1]), 2, (0, 255, 255), 1,
#            cv2.LINE_AA)

# # nose x
# nose_x_axis_k = (mesh_points[278][1] - mesh_points[48][1]) / (
#         mesh_points[278][0] - mesh_points[48][0])
# nose_x_axis_b = mesh_points[48][1] - (nose_x_axis_k * mesh_points[48][0])
# nose_x_axis_y1 = int(nose_x_axis_b)
# nose_x_axis_y2 = int(detected_face.shape[1] * nose_x_axis_k + nose_x_axis_b)
# cv2.line(detected_face, (0, nose_x_axis_y1),
#          (detected_face.shape[1], nose_x_axis_y2), (0, 0, 255), 1, cv2.LINE_AA)

# # nose y axis
# if mesh_points[200][0] - mesh_points[151][0] != 0:
#     nose_y_axis_k = (mesh_points[200][1] - mesh_points[151][1]) / (
#             mesh_points[200][0] - mesh_points[151][0])
#     nose_y_axis_b = mesh_points[151][1] - (nose_y_axis_k * mesh_points[151][0])
#     nose_y_axis_y1 = int(nose_y_axis_b)
#     nose_y_axis_y2 = int(detected_face.shape[1] * nose_y_axis_k + nose_y_axis_b)
#     cv2.line(detected_face, (0, nose_y_axis_y1),
#              (detected_face.shape[1], nose_y_axis_y2), (0, 255, 0), 1, cv2.LINE_AA)
#     # cv2.line(detected_face, (mesh_points[151][0], mesh_points[151][1]),
#     #          (mesh_points[200][0], mesh_points[200][1]), (0, 255, 0), 1, cv2.LINE_AA)
#
#     # пересечение
#     a = np.array([[1, -nose_x_axis_k], [1, -nose_y_axis_k]])
#     b = np.array([nose_x_axis_b, nose_y_axis_b])
#     nose_x_axis_center = int(np.linalg.solve(a, b)[1])
#     nose_y_axis_center = int(np.linalg.solve(a, b)[0])
#     print(nose_x_axis_center, nose_y_axis_center)
#     cv2.circle(detected_face, (nose_x_axis_center, nose_y_axis_center), 2, (255, 255, 255), 2,
#                cv2.LINE_AA)
#
#     # вектор носа
#     cv2.line(detected_face, (mesh_points[4][0], mesh_points[4][1]),
#              (nose_x_axis_center, nose_y_axis_center), (255, 0, 0), 1, cv2.LINE_AA)
#
# # nose
# cv2.circle(detected_face, (mesh_points[4][0], mesh_points[4][1]), 2, (255, 0, 0), 2,
#            cv2.LINE_AA)
#
# shift_x = mesh_points[4][0] - nose_x_axis_center
# shift_y = mesh_points[4][1] - nose_y_axis_center
#
# # cv2.line(detected_face, (mesh_points[48][0], mesh_points[48][1]),
# #          (mesh_points[278][0], mesh_points[278][1]), (0, 0, 255), 1, cv2.LINE_AA)
# nose_x_center = int(mesh_points[48][0] + (mesh_points[278][0] - mesh_points[48][0]) / 2)
# nose_x_center_y = int(mesh_points[48][1] + (mesh_points[278][1] - mesh_points[48][1]) / 2)
# cv2.line(detected_face,
#          (mesh_points[151][0], mesh_points[151][1]),
#          (nose_x_center, nose_x_center_y),
#         (0, 255, 0), 1, cv2.LINE_AA)
# cv2.line(detected_face,
#          (int(mesh_points[69][0] + ((mesh_points[299][0] - mesh_points[69][0]) / 2)), 0),
#          (int(mesh_points[69][0] + ((mesh_points[299][0] - mesh_points[69][0]) / 2)),
#           detected_face.shape[0]), (255, 255, 255), 1, cv2.LINE_AA)

# cv2.circle(detected_face, (mesh_points[10][0], mesh_points[10][1]), 2, (255, 255, 255), 1,
#            cv2.LINE_AA)
# cv2.circle(detected_face, (mesh_points[152][0], mesh_points[152][1]), 2, (255, 255, 255), 1,
#            cv2.LINE_AA)
#
# cv2.circle(detected_face, (mesh_points[299][0], mesh_points[299][1]), 2, (0, 255, 255), 1,
#            cv2.LINE_AA)
# cv2.circle(detected_face, (mesh_points[69][0], mesh_points[69][1]), 2, (0, 255, 255), 1,
#            cv2.LINE_AA)
# # cv2.line(detected_face, (0, mesh_points[69][1]), (detected_face.shape[1], mesh_points[299][1]), (0, 255, 255), 2, cv2.LINE_AA)
# cv2.line(detected_face, (mesh_points[69][0], mesh_points[69][1]), (mesh_points[299][0], mesh_points[299][1]), (0, 255, 255), 1, cv2.LINE_AA)
#
#
# # x axis red
# x_axis_k = (mesh_points[299][1] - mesh_points[69][1]) / (mesh_points[299][0] - mesh_points[69][0])
# x_axis_b = mesh_points[69][1] - (x_axis_k * mesh_points[69][0])
# x_axis_y1 = int(x_axis_b)
# x_axis_y2 = int(detected_face.shape[1] * x_axis_k + x_axis_b)
# x_axis_len = math.sqrt((detected_face.shape[1])**2 + (x_axis_y2 - x_axis_y1)**2)
# x_axis_sin = (x_axis_y2 - x_axis_y1) / x_axis_len
# x_axis_cos = math.sqrt(1 - x_axis_sin**2)
# cv2.line(detected_face, (0, 30),
#          (detected_face.shape[1], int(30*x_axis_cos)),
#          (0, 0, 255), 1, cv2.LINE_AA)
#
# cv2.line(detected_face, (0, x_axis_y1),
#          (detected_face.shape[1], x_axis_y2),
#          (0, 0, 255), 1, cv2.LINE_AA)
#
# center_x_axis = int(mesh_points[69][0] + ((mesh_points[299][0] - mesh_points[69][0]) / 2))

# y axis green
# cv2.line(detected_face, (center_x_axis * math.c, 0),
#          (detected_face.shape[1], x_axis_y2),
#          (0, 0, 255), 1, cv2.LINE_AA)
# z axis blue

# center_l = np.array([cX_l, cY_l], dtype=np.int32)
# # cv2.circle(detected_face, center_l, int(rad_l), (255, 255, 255), 1, cv2.LINE_AA)
#
# center_r = np.array([cX_r, cY_r], dtype=np.int32)
# cv2.circle(detected_face, center_r, int(rad_r), (255, 255, 255), 1, cv2.LINE_AA)

# cv2.polylines(detected_face, [np.array([mesh_points[p] for p in LEFT_EYE], dtype=np.int32)], True,
#                       (0, 255, 0),
#                       1, cv2.LINE_AA)
# cv2.polylines(detected_face, [np.array([mesh_points[p] for p in RIGHT_EYE], dtype=np.int32)], True,
#                       (0, 255, 0),
#                       1, cv2.LINE_AA)

# left_eye_coords = [mesh_points[p] for p in LEFT_EYE]
# l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
# l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
# l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
# l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]
# l_width = l_max_x - l_min_x
# l_height = l_max_y - l_min_y
# l_center_x = int(l_min_x + (l_width / 2))
# l_center_y = int(l_min_y + (l_height / 2))
#
# right_eye_coords = [mesh_points[p] for p in RIGHT_EYE]
# r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
# r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
# r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
# r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]
# r_width = r_max_x - r_min_x
# r_height = r_max_y - r_min_y
# r_center_x = int(r_min_x + (r_width / 2))
# r_center_y = int(r_min_y + (r_height / 2))
#
# left_ratio_h = (cX_l - l_min_x) / l_width
# left_ratio_v = (cY_l - l_min_y) / l_height
# right_ratio_h = (cX_r - r_min_x) / r_width
# right_ratio_v = (cY_r - r_min_y) / r_height
#
# # cX_r -= shift_x
# # cX_l -= shift_x
# # cY_r -= shift_y
# # cY_l -= shift_y
#
# # if abs(r_center_x - old_r_center_x) >= EPSILLON and abs(r_center_y - old_r_center_y) >= EPSILLON:
# #     old_r_center_x = r_center_x
# #     old_r_center_y = r_center_y
# # else:
# #     r_center_x = old_r_center_x
# #     r_center_y = old_r_center_y
# #
# # if abs(cX_r - old_cX_r) >= EPSILLON and abs(cY_r - old_cY_r) >= EPSILLON:
# #     old_cX_r = cX_r
# #     old_cY_r = cY_r
# # else:
# #     cX_r = old_cX_r
# #     cY_r = old_cY_r
# cv2.line(detected_face, (r_center_x - shift_x, r_center_y - shift_y),
#          (int(cX_r), int(cY_r)), (255, 255, 0), 2, cv2.LINE_AA)
# cv2.line(detected_face, (l_center_x - shift_x, l_center_y - shift_y),
#          (int(cX_l), int(cY_l)), (255, 255, 0), 2, cv2.LINE_AA)

# l
# if abs(l_center_x - old_l_center_x) > EPSILLON:
#     old_l_center_x = l_center_x
# else:
#     l_center_x = old_l_center_x
#
# if abs(cX_l - old_cX_l) > EPSILLON:
#     old_cX_l = cX_l
# else:
#     cX_l = old_cX_l

# r_center_x -= shift_x
# l_center_x -= shift_x
# r_center_y -= shift_y
# l_center_y -= shift_y
#
# try:
#     tmp_k_r = (r_center_y - cY_r) / (r_center_x - cX_r)
#     tmp_b_r = cY_r - (tmp_k_r * cX_r)
# except:
#     ...
# else:
#     try:
#         # cv2.line(detected_face, (int(r_min_x + (r_width / 2)), int(r_min_y + (r_height / 2))),
#         #          (int(100), int(tmp_k_r*100 + tmp_b_r)), (255, 255, 0), 2, cv2.LINE_AA)
#
#         if r_center_y > cY_r:
#             cv2.line(detected_face, (r_center_x, r_center_y),
#                      (int((0 - tmp_b_r) / tmp_k_r), int(0)), (255, 0, 255), 1, cv2.LINE_AA)
#         else:
#             cv2.line(detected_face, (r_center_x, r_center_y),
#                      (int((100 - tmp_b_r) / tmp_k_r), int(100)), (255, 0, 255), 1, cv2.LINE_AA)
#     except:
#         ...
#
# try:
#     tmp_k_l = (l_center_y - cY_l) / (l_center_x - cX_l)
#     tmp_b_l = cY_l - (tmp_k_l * cX_l)
# except:
#     ...
# else:
#     try:
#         if l_center_y > cY_l:
#             cv2.line(detected_face, (l_center_x, l_center_y),
#                      (int((0 - tmp_b_l) / tmp_k_l), int(0)), (255, 0, 255), 1, cv2.LINE_AA)
#         else:
#             cv2.line(detected_face, (l_center_x, l_center_y),
#                      (int((100 - tmp_b_l) / tmp_k_l), int(100)), (255, 0, 255), 1, cv2.LINE_AA)
#
#     except:
#         ...
# cv2.imshow('detected face', detected_face)
