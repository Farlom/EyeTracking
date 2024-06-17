from __future__ import division

import cv2
import numpy as np
import mediapipe as mp
from mp_eye import Eye
from calibration import Calibration
import settings
import math
from statistics import mean

from l2cs import Pipeline, render
import pathlib
import torch


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    EPSILLON = 5
    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self, detector, shape):
        self.frame = None
        self.num_faces = 0
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.face = None
        self.face_frame = None

        self.left_pupil = None
        self.right_pupil = None
        self.detector = detector

        self.width, self.height = shape

        self.x_left_old = 0
        self.y_left_old = 0
        self.landmarks = None
        self.pts = []
        self.distance = None
        self.updown_anle = None
        self.__pitch = None
        self.yaw = None
        self.__roll = None
        self.vector = None
        self.point_2d = None
        # _face_detector is used to detect faces
        self.__face_detector = mp.solutions.face_detection.FaceDetection()
        # _predictor is used to get facial landmarks of a given face
        self.__face_landmarker = mp.solutions.face_mesh.FaceMesh(refine_landmarks=False)

        self.points = []
        self.CWD = pathlib.Path.cwd()
        self.gaze_pipeline = Pipeline(
            weights= self.CWD / 'models' / 'model_gaze360.pkl',
            arch='ResNet50',
            device=torch.device('cpu')  # or 'gpu'
        )

        self.tn = 0
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.row = 0
        self.col = 0

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(frame)

        if result.detections:
            self.num_faces = len(result.detections)
            # print(result.detections)
            if self.num_faces == 1:
                result_faces = result.detections[0].location_data.relative_bounding_box

                # epsilon = 10
                epsilon = settings.BBOX_EPSILON
                # bbox_delta = 50
                bbox_delta = settings.BBOX_DELTA
                x1 = round(result_faces.xmin * self.width) - bbox_delta
                x1 = x1 if x1 >= 0 else 0

                y1 = round(result_faces.ymin * self.height) - bbox_delta
                y1 = y1 if y1 >= 0 else 0

                x2 = round((result_faces.xmin + result_faces.width) * self.width) + bbox_delta
                x2 = x2 if x2 <= self.width else self.width

                y2 = round((result_faces.ymin + result_faces.height) * self.height) + bbox_delta
                y2 = y2 if y2 <= self.height else self.height



                if settings.ENABLE_SMOOTH_BBOX_RENDER:
                    if self.face and abs(self.face[0][0] - x1) <= epsilon:
                        x1 = self.face[0][0]
                    if self.face and abs(self.face[0][1] - y1) <= epsilon:
                        y1 = self.face[0][1]
                    if self.face and abs(self.face[1][0] - x2) <= epsilon:
                        x2 = self.face[1][0]
                    if self.face and abs(self.face[1][1] - y2) <= epsilon:
                        y2 = self.face[1][1]


                faces = [(x1, y1), (x2, y2)]
                self.face = faces

                if settings.SHOW_FACE_RECTANGLE_2D:
                    cv2.rectangle(frame, faces[0], faces[1], (0, 255, 0), 1)
                    # pass
                face_frame = frame[y1:y2, x1:x2]
                self.face_frame = face_frame
            elif self.num_faces == 2:
                distance = self.width
                for i in range(self.num_faces):
                    result_faces = result.detections[i].location_data.relative_bounding_box

                    epsilon = settings.BBOX_EPSILON

                    x1 = round(result_faces.xmin * self.width)
                    y1 = round(result_faces.ymin * self.height)
                    x2 = round((result_faces.xmin + result_faces.width) * self.width)
                    y2 = round((result_faces.ymin + result_faces.height) * self.height)

                    if settings.ENABLE_SMOOTH_BBOX_RENDER:
                        if self.face and abs(self.face[0][0] - x1) <= epsilon:
                            x1 = self.face[0][0]
                        if self.face and abs(self.face[0][1] - y1) <= epsilon:
                            y1 = self.face[0][1]
                        if self.face and abs(self.face[1][0] - x2) <= epsilon:
                            x2 = self.face[1][0]
                        if self.face and abs(self.face[1][1] - y2) <= epsilon:
                            y2 = self.face[1][1]

                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                    if x1 + x2 // 2 < distance:
                        distance = x1 + x2 // 2
                        faces = [(x1, y1), (x2, y2)]
                        self.face = faces
                        face_frame = frame[y1:y2, x1:x2]
                        self.face_frame = face_frame

            # cv2.imshow('f',face_frame)
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=settings.MEDIAPIPE_PUPIL_DETECTION,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as mesh:

                # print(mesh.process(self.face_frame))
                # input()
                res_landmarks = mesh.process(self.face_frame).multi_face_landmarks
                # print(res_landmarks)
                if res_landmarks:
                    face_height, face_width = self.face_frame.shape[:2]


                    landmarks = res_landmarks[0].landmark
                    landmarks = np.array(
                        [np.multiply([p.x, p.y], [face_width, face_height]).astype(int) for p in
                         landmarks]
                    )

                    if type(self.landmarks) != type(None):
                        for i in range(len(landmarks)):
                            if abs(landmarks[i][0] - self.landmarks[i][0]) >= self.EPSILLON // 2 or abs(
                                    landmarks[i][1] - self.landmarks[i][1]) >= self.EPSILLON // 2:
                                self.landmarks = landmarks
                            else:
                                landmarks = self.landmarks
                    self.landmarks = landmarks
                    self.eye_left = Eye(self.face_frame, landmarks, 0, self.calibration)
                    self.eye_right = Eye(self.face_frame, landmarks, 1, self.calibration)
                else:
                    self.landmarks = None
        else:
            self.num_faces = 0

    def refresh(self, frame: np.ndarray) -> None:
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x + self.face[0][0]
            y = self.eye_left.origin[1] + self.eye_left.pupil.y + self.face[0][1]
            cx = self.eye_left.origin[0] + round(self.eye_left.center[0]) + self.face[0][0]
            cy = self.eye_left.origin[1] + round(self.eye_left.center[1]) + self.face[0][1]

            self.left_pupil = (x, y)
            return x, y, cx, cy

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x + self.face[0][0]
            y = self.eye_right.origin[1] + self.eye_right.pupil.y + self.face[0][1]
            cx = self.eye_right.origin[0] + round(self.eye_right.center[0]) + self.face[0][0]
            cy = self.eye_right.origin[1] + round(self.eye_right.center[1]) + self.face[0][1]

            self.right_pupil = (x, y)
            return x, y, cx, cy

    def __estimate_distance(self, img):
        left_eye_center_x, left_eye_center_y = self.pupil_left_coords()[2:]
        right_eye_center_x, right_eye_center_y = self.pupil_right_coords()[2:]

        # actual distance between eyes in px
        w = math.sqrt((right_eye_center_x - left_eye_center_x) ** 2 + (right_eye_center_y - left_eye_center_y) ** 2)
        # print(w)

        # average distance between eyes in cm with the d = 50
        W = 6.3
        d = 50
        # print((w * d) / W)
        f = 1000
        d = (W * f) / w
        # print(d)
        cv2.putText(img, f'distance (cm): {d:.3f}', (300, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
        cv2.putText(img, f'distance (cm): {d:.3f}', (300, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        self.distance = d

    def get_2d_points(self, img, rotation_vector, translation_vector, camera_matrix, val):
        # https://github.com/vardanagarwal/Proctoring-AI/blob/master/head_pose_estimation.py#L44
        """Return the 3D points present as 2D for making annotation box"""
        point_3d = []
        dist_coeffs = np.zeros((4, 1))
        rear_size = val[0]
        rear_depth = val[1]
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = val[2]
        front_depth = val[3]
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          camera_matrix,
                                          dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))


        y = (point_2d[5] + point_2d[8]) // 2
        x = point_2d[2]

        # x1 = x
        # x2 = y
        # alpha = (x1[1] - x1[1])/(x2[0] - x1[0])
        # angle2 = int(math.degrees(math.atan(-1/alpha)))
        # cv2.putText(img, f'Rotate: {angle2}', (750, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


        return point_2d

    def __3d_to_2d(self, img, side):
        NOSE_TIP = (self.face[0][0] + self.landmarks[4][0], self.face[0][1] + self.landmarks[4][1])
        CHIN = (self.face[0][0] + self.landmarks[152][0], self.face[0][1] + self.landmarks[152][1])
        LEFT_EYE_CORNER = (self.face[0][0] + self.landmarks[263][0], self.face[0][1] + self.landmarks[263][1])
        RIGHT_EYE_CORNER = (self.face[0][0] + self.landmarks[33][0], self.face[0][1] + self.landmarks[33][1])
        LEFT_MOUTH_CORNER = (self.face[0][0] + self.landmarks[287][0], self.face[0][1] + self.landmarks[287][1])
        RIGHT_MOUTH_CORNER = (self.face[0][0] + self.landmarks[57][0], self.face[0][1] + self.landmarks[57][1])

        NOSE_TIP_T = (self.face[0][0] + self.landmarks[4][0], self.face[0][1] + self.landmarks[4][1], 0)
        CHIN_T = (self.face[0][0] + self.landmarks[152][0], self.face[0][1] + self.landmarks[152][1], 0)
        LEFT_EYE_CORNER_T = (self.face[0][0] + self.landmarks[263][0], self.face[0][1] + self.landmarks[263][1], 0)
        RIGHT_EYE_CORNER_T = (self.face[0][0] + self.landmarks[33][0], self.face[0][1] + self.landmarks[33][1], 0)
        LEFT_MOUTH_CORNER_T = (self.face[0][0] + self.landmarks[287][0], self.face[0][1] + self.landmarks[287][1], 0)
        RIGHT_MOUTH_CORNER_T = (self.face[0][0] + self.landmarks[57][0], self.face[0][1] + self.landmarks[57][1], 0)

        EYE_BALL_CENTER_LEFT = np.array([[29.05], [32.7], [-39.5]])
        EYE_BALL_CENTER_RIGHT = np.array([[-29.05], [32.7], [-39.5]])
        EYE_BALL_CENTER_MEAN = np.array([[0], [32.7], [-39.5]])

        image_points = np.array([
            NOSE_TIP,
            CHIN,
            LEFT_EYE_CORNER,
            RIGHT_EYE_CORNER,
            LEFT_MOUTH_CORNER,
            RIGHT_MOUTH_CORNER
        ], dtype='double')


        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            NOSE_TIP_T,
            CHIN_T,
            LEFT_EYE_CORNER_T,
            RIGHT_EYE_CORNER_T,
            LEFT_MOUTH_CORNER_T,
            RIGHT_MOUTH_CORNER_T
        ], dtype="double")
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])



        '''
                    camera matrix estimation
                    '''
        focal_length = img.shape[1]
        center = (img.shape[1] / 2, img.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        '''
        3D model eye points
        The center of the eye ball
        '''
        if side == 0:  # left
            # 2d pupil location
            x_left, y_left, _, _ = self.pupil_left_coords()

            # x_left += self.face[0][0]
            # y_left += self.face[0][1]
            pupil = self.left_pupil
            # Transformation between image point to world point
            _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

            if transformation is not None:  # if estimateAffine3D secsseded
                # project pupil image point into 3d world point
                pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T

                # 3D gaze point (10 is arbitrary value denoting gaze distance)
                S = EYE_BALL_CENTER_LEFT + (pupil_world_cord - EYE_BALL_CENTER_LEFT) * 10

                # Project a 3D gaze direction onto the image plane.
                (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

                # project 3D head pose into the image plane
                (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                                   rotation_vector,
                                                   translation_vector, camera_matrix, dist_coeffs)

                # correct gaze for head rotation
                gaze = pupil + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)

                # Draw gaze line into screen
                p1 = (int(pupil[0]), int(pupil[1]))
                p2 = (int(gaze[0]), int(gaze[1]))
                cv2.line(img, p1, p2, (0, 0, 255), 2)
                return (p1, p2)
        elif side == 1:
            # 2d pupil location
            x_right, y_right, _, _ = self.pupil_right_coords()
            # x_right += self.face[0][0]
            # y_right += self.face[0][1]

            pupil = (x_right, y_right)

            # Transformation between image point to world point
            _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

            if transformation is not None:  # if estimateAffine3D secsseded
                # project pupil image point into 3d world point
                pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T

                # 3D gaze point (10 is arbitrary value denoting gaze distance)
                S_right = EYE_BALL_CENTER_RIGHT + (pupil_world_cord - EYE_BALL_CENTER_RIGHT) * 10

                # Project a 3D gaze direction onto the image plane.

                (eye_pupil2D, _) = cv2.projectPoints((int(S_right[0]), int(S_right[1]), int(S_right[2])),
                                                     rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
                # project 3D head pose into the image plane
                (head_pose, _) = cv2.projectPoints(
                    (int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                    rotation_vector,
                    translation_vector, camera_matrix, dist_coeffs)
                # correct gaze for head rotation
                gaze = pupil + (eye_pupil2D[0][0] - pupil) - (
                        head_pose[0][0] - pupil)

                # Draw gaze line into screen
                p1 = (int(pupil[0]), int(pupil[1]))
                p2 = (int(gaze[0]), int(gaze[1]))
                # print(S_right)
                # cv2.circle(img, (round(eye_pupil2D[0][0][0]), round(eye_pupil2D[0][0][1])), 3, (255, 255, 255), 2)
                cv2.line(img, p1, p2, (0, 0, 255), 2)

                return (p1, p2)
        else:
            # 2d pupil location
            x_right, y_right, _, _ = self.pupil_right_coords()
            x_left, y_left = self.left_pupil

            pupil = (round(mean([x_right, x_left])), round(mean([y_right, y_left])))

            # Transformation between image point to world point
            _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

            if transformation is not None:  # if estimateAffine3D secsseded
                # project pupil image point into 3d world point
                pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T

                # 3D gaze point (10 is arbitrary value denoting gaze distance)
                S = EYE_BALL_CENTER_MEAN + (pupil_world_cord - EYE_BALL_CENTER_MEAN) * int(self.distance // 10)
                # print(S)

                # Project a 3D gaze direction onto the image plane.

                (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])),
                                                     rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
                # project 3D head pose into the image plane
                (head_pose, _) = cv2.projectPoints(
                    (int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                    rotation_vector,
                    translation_vector, camera_matrix, dist_coeffs)
                # correct gaze for head rotation
                gaze = pupil + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)

                # Draw gaze line into screen
                p1 = (int(pupil[0]), int(pupil[1]))
                p2 = (int(gaze[0]), int(gaze[1]))
                cv2.line(img, p1, p2, (0, 0, 255), 2)
                self.vector = p2
                return (p1, p2)

    def __estimate(self, frame: np.ndarray, output: bool):
        # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        NOSE_TIP = (self.face[0][0] + self.landmarks[4][0], self.face[0][1] + self.landmarks[4][1])
        CHIN = (self.face[0][0] + self.landmarks[152][0], self.face[0][1] + self.landmarks[152][1])
        LEFT_EYE_CORNER = (self.face[0][0] + self.landmarks[130][0], self.face[0][1] + self.landmarks[130][1])
        RIGHT_EYE_CORNER = (self.face[0][0] + self.landmarks[359][0], self.face[0][1] + self.landmarks[359][1])
        LEFT_MOUTH_CORNER = (self.face[0][0] + self.landmarks[61][0], self.face[0][1] + self.landmarks[61][1])
        RIGHT_MOUSE_CORNER = (self.face[0][0] + self.landmarks[291][0], self.face[0][1] + self.landmarks[291][1])

        image_points = np.array([
            NOSE_TIP,
            CHIN,
            LEFT_EYE_CORNER,
            RIGHT_EYE_CORNER,
            LEFT_MOUTH_CORNER,
            RIGHT_MOUSE_CORNER
        ], dtype='double')

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # model_points = np.array([
        #     (0.0, 0.0, 0.0),  # Nose tip
        #     (0, -63.6, -12.5),  # Chin
        #     (-43.3, 32.7, -26),  # Left eye, left corner
        #     (43.3, 32.7, -26),  # Right eye, right corner
        #     (-28.9, -28.9, -24.1),  # Left Mouth corner
        #     (28.9, -28.9, -24.1)  # Right mouth corner
        # ])

        # *Центр левого глаза: (-135, 170, -135)
        # *Центр правого глаза: (135, 170, -135

        focal_length = self.width
        center = (self.width / 2, self.height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs)
        # if output:
        print("Camera Matrix :\n {0}".format(camera_matrix))
        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        x_right, y_right, x_right_center, y_right_center = self.pupil_right_coords()
        # x_right += self.face[0][0]
        # y_right += self.face[0][1]

        if settings.SHOW_NOSE_RAY:
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 400.0)]), rotation_vector,
                                                             translation_vector,
                                                             camera_matrix, dist_coeffs)

            nose_p1 = (int(image_points[0][0]), int(image_points[0][1]))
            nose_p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(frame, nose_p1, nose_p2, (255, 0, 0), 2)

        if settings.SHOW_LEFT_GAZE_RAY:
            (x_left_circle, y_left_circle), rad_left_circle = cv2.minEnclosingCircle(
                self.landmarks[self.LEFT_EYE_POINTS])
            x_left_circle = int(x_left_circle + self.face[0][0])
            y_left_circle = int(y_left_circle + self.face[0][1])

            (left_end_point2D, jacobian) = cv2.projectPoints(np.array([(-135.0, 170.0, 400.0)]), rotation_vector,
                                                             translation_vector,
                                                             camera_matrix, dist_coeffs)

            p1 = (x_left_circle, y_left_circle)
            p2 = (int(left_end_point2D[0][0][0]), int(left_end_point2D[0][0][1]))
            cv2.line(frame, self.left_pupil, p2, (255, 0, 0), 2)

        if settings.SHOW_RIGHT_GAZE_RAY:
            # (x_right_circle, y_right_circle), rad_right_circle = cv2.minEnclosingCircle(
            #     self.landmarks[self.RIGHT_EYE_POINTS])
            # x_right_circle = int(x_right_circle + self.face[0][0])
            # y_right_circle = int(y_right_circle + self.face[0][1])
            (right_end_point2D, jacobian) = cv2.projectPoints(np.array([(135.0, 170.0, 400.0)]), rotation_vector,
                                                              translation_vector,
                                                              camera_matrix, dist_coeffs)
            # p1 = (x_right_circle, y_right_circle)
            p2 = (int(right_end_point2D[0][0][0]), int(right_end_point2D[0][0][1]))
            cv2.line(frame, self.right_pupil, p2, (255, 255, 255), 2)

        if settings.SHOW_MEAN_GAZE_RAY:
            x_right, y_right = self.right_pupil
            x_left, y_left = self.left_pupil

            pupil = (round(mean([x_right, x_left])), round(mean([y_right, y_left])))
            (right_end_point2D, jacobian) = cv2.projectPoints(np.array([(0, 170.0, 600.0)]), rotation_vector,
                                                              translation_vector,
                                                              camera_matrix, dist_coeffs)
            # ry = y cos α + x sin α
            p2 = (int(right_end_point2D[0][0][0]), int(right_end_point2D[0][0][1]))
            cv2.line(frame, pupil, p2, (255, 255, 255), 2)
        rear_size = 300
        rear_depth = 0
        front_size = 500
        front_depth = 500
        point_2d = self.get_2d_points(frame, rotation_vector, translation_vector, camera_matrix, [rear_size, rear_depth, front_size, front_depth])
        self.point_2d = point_2d
        #https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV

        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        # print(math.sin(pitch))
        pitch = int(math.degrees(math.asin(math.sin(pitch))))
        self.__pitch = pitch

        roll = int(-math.degrees(math.asin(math.sin(roll))))
        self.__roll = roll

        yaw = int(math.degrees(math.asin(math.sin(yaw))))
        self.__yaw = yaw
        if settings.SHOW_TEXT:
            cv2.putText(frame, f'pitch: {pitch}', (self.face[1][0], self.face[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f'pitch: {pitch}', (self.face[1][0], self.face[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            cv2.putText(frame, f'roll: {roll}', (self.face[1][0], self.face[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f'roll: {roll}', (self.face[1][0], self.face[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            cv2.putText(frame, f'yaw: {yaw}', (self.face[1][0], self.face[0][1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f'yaw: {yaw}', (self.face[1][0], self.face[0][1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        if settings.SHOW_FACE_RECTANGLE_3D:
            # inner
            cv2.line(frame, point_2d[0], point_2d[1], (255, 255, 0), 2)
            cv2.line(frame, point_2d[1], point_2d[2], (255, 255, 0), 2)
            cv2.line(frame, point_2d[2], point_2d[3], (255, 255, 0), 2)
            cv2.line(frame, point_2d[3], point_2d[4], (255, 255, 0), 2)

            # rebra
            cv2.line(frame, point_2d[1], point_2d[6], (255, 255, 200), 2)
            cv2.line(frame, point_2d[2], point_2d[7], (255, 255, 200), 2)
            cv2.line(frame, point_2d[3], point_2d[8], (255, 255, 200), 2)
            cv2.line(frame, point_2d[4], point_2d[9], (255, 255, 200), 2)

            # outer
            cv2.line(frame, point_2d[5], point_2d[6], (255, 255, 255), 3) # 5 лево низ ; 6 лево верх
            cv2.line(frame, point_2d[6], point_2d[7], (255, 255, 255), 3) # 7 право верх 8 право низ
            cv2.line(frame, point_2d[7], point_2d[8], (255, 255, 255), 3)
            cv2.line(frame, point_2d[8], point_2d[9], (255, 255, 255), 3)
            # cv2.circle(frame, point_2d[9], 5, (0, 0, 0), -1)

    def l2cs(self, frame, face_frame):
        result = self.gaze_pipeline.step(face_frame)
        print(result)
        frame = render(frame[
                       self.face[0][1]:self.face[0][1] + self.face[1][1],
                       self.face[0][0]:self.face[0][0] + self.face[1][0]
                       ], result)
        # cv2.imshow('face', face_frame)
        return result
    def l2cs_draw_face(self, frame, result):
        x_min = result.bboxes[0][0]
        x_max = result.bboxes[0][2]
        y_min = result.bboxes[0][1]
        y_max = result.bboxes[0][3]

        face_width = x_max - x_min
        face_height = y_max - y_min

        cv2.rectangle(frame, (int(self.face[0][0] + x_min), int(self.face[0][1] + y_min)),
                  (int(self.face[0][0] + x_max), int(self.face[0][1] + y_max)),
                  (255, 255, 255), 1)

    def l2cs_draw_gaze(self, frame, result):
        x_min = result.bboxes[0][0]
        x_max = result.bboxes[0][2]
        y_min = result.bboxes[0][1]
        y_max = result.bboxes[0][3]

        face_width = x_max - x_min
        face_height = y_max - y_min

        arrow_length = self.width * 0.5
        dx = -arrow_length * np.sin(result.pitch) * np.cos(result.yaw)
        dy = -arrow_length * np.sin(result.yaw)
        cv2.arrowedLine(
            frame,
            (int(self.face[0][0] + x_min + face_width / 2), int(self.face[0][1] + y_min + face_height / 2)),
            (int(self.face[0][0] + x_min + dx + face_width / 2), int(self.face[0][1] + y_min + face_height / 2 + dy)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )
    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()
        epsilon = 0
        cv2.putText(frame, f'num faces: {self.num_faces}', (1200, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
        cv2.putText(frame, f'num faces: {self.num_faces}', (1200, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)

        if settings.SHOW_SCREEN_GRID:
            for i in range(settings.SCREEN_H - 1):
                cv2.line(frame, (0, self.height // settings.SCREEN_H * (i + 1)),
                         (self.width, self.height // settings.SCREEN_H * (i + 1)), (255, 255, 0), 1)
            for i in range(settings.SCREEN_W - 1):
                cv2.line(frame, (self.width // settings.SCREEN_W * (i + 1), 0),
                         (self.width // settings.SCREEN_W * (i + 1), self.height), (255, 255, 0), 1)

        # counter = 0
        # if settings.EVALUATION:
        if self.num_faces > 0:
            if self.landmarks is None:
                cv2.putText(frame, f'CANNOT DETECT LANDMARKS', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
                cv2.putText(frame, f'CANNOT DETECT LANDMARKS', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                self.fn += 1
            elif self.landmarks is not None and not self.pupils_located:
                self.fn += 1
                cv2.putText(frame, f'CANNOT DETECT PUPILS', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
                cv2.putText(frame, f'CANNOT DETECT PUPILS', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        if self.pupils_located:
            # self.tp += 1
            color = (0, 255, 0)
            x_left, y_left, x_left_center, y_left_center = self.pupil_left_coords()
            x_right, y_right, x_right_center, y_right_center = self.pupil_right_coords()


            # estimation
            if self.landmarks is not None:
                self.__estimate(frame, False)

            # ptrs
            if settings.SHOW_PUPIL_POINTERS:
                if self.landmarks is not None and self.pupils_located:
                    cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
                    cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
                    cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
                    cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)



            # face rect
            if settings.SHOW_FACE_RECTANGLE_2D:
                cv2.rectangle(frame, (self.face[0][0], self.face[0][1]), (self.face[1][0], self.face[1][1]), color)

            # frame = self.frame.copy()
            self.__estimate_distance(frame)

            # if settings.SHOW_LEFT_RAY and self.landmarks is not None:
            #     self.__3d_to_2d(frame, 0)
            #
            # if settings.SHOW_RIGHT_RAY and self.landmarks is not None:
            #     self.__3d_to_2d(frame, 1)
            #
            # if settings.SHOW_MEAN_RAY and self.landmarks is not None:
            #     # self.vector = (
            #     self.__3d_to_2d(frame, 2)

            # for landmark in self.landmarks:
            #     cv2.circle(frame, (self.face[0][0] + landmark[0], self.face[0][1] + landmark[1]), 1, (255, 255, 255), -1)

            # cv2.circle(frame,
            #            (self.face[0][0] + self.landmarks[473][0],
            #             self.face[0][1] + self.landmarks[473][1]), 2, (0, 255, 255), -1)
            #
            # cv2.circle(frame,
            #            (self.face[0][0] + self.landmarks[468][0],
            #             self.face[0][1] + self.landmarks[468][1]), 2, (0, 255, 255), -1)
            # cv2.circle(frame, (frame.shape[1] //2, frame.shape[0] // 2), 3, (255, 255, 255), -1)
            # cv2.rectangle(frame, (100, 100), (frame.shape[1] - 100, frame.shape[0] - 100), (255, 0,0), 1)
            # print(self.face_frame is None)
            # result = self.gaze_pipeline.step(self.face_frame)

            # l2cs
            try:
                print(self.face_frame is None)
                # result = self.l2cs(frame, self.face_frame)
                result = self.gaze_pipeline.step(self.face_frame)
                # print(result)
                if settings.SHOW_L2CS_RECTANGLE:
                    self.l2cs_draw_face(frame, result)
                if settings.SHOW_GAZE_RAY:
                    self.l2cs_draw_gaze(frame, result)
                if True:
                    DISTANCE_TO_OBJECT = self.distance * 10 # * 1.5 # mm
                    # HEIGHT_OF_HUMAN_FACE = 0.25 * self.distance * 10 # 250  # mm
                    HEIGHT_OF_HUMAN_FACE = 250
                    CAMERA_ANGLE = math.radians(self.distance / 10) if self.distance < 1140 else 0# 10
                    face_pitch = math.radians(self.__pitch)
                    face_yaw = math.radians(self.__yaw)
                    length_per_pixel = HEIGHT_OF_HUMAN_FACE / (result.bboxes[0][3] - result.bboxes[0][1])

                    gaze_pitch = result.yaw[0]
                    # gaze_pitch = result.yaw[0] / math.radians(10) * math.radians(30)

                    gaze_yaw = result.pitch[0]
                    # gaze_yaw = result.pitch[0] / math.radians(20) * math.radians(50)
                    #
                    pitch = CAMERA_ANGLE + face_pitch + gaze_pitch
                    yaw = face_yaw + gaze_yaw

                    if gaze_pitch <= face_pitch:
                        dy = -DISTANCE_TO_OBJECT * np.arccos(gaze_yaw) * np.tan(gaze_pitch) / length_per_pixel
                    else:
                        dy = -DISTANCE_TO_OBJECT * np.arccos(gaze_yaw) * np.tan(gaze_pitch - face_pitch) / length_per_pixel
                    # dy = -DISTANCE_TO_OBJECT * np.arccos(result.pitch) * np.tan(result.yaw) / length_per_pixel
                    # dy = 0
                    dy = -DISTANCE_TO_OBJECT * np.arccos(yaw) * np.tan(gaze_pitch) / length_per_pixel
                    dy = -self.distance * 10 * np.sin(result.yaw)

                    dy = dy if not np.isnan(dy) else 100000000
                    dx = -self.distance * 10 * np.sin(result.pitch + face_yaw) #* np.cos(result.yaw)


                    gaze_point = (int(self.width / 2 + dx), int(self.height / 2 + dy))
                                  # int(self.face[0][1] + result.bboxes[0][1] + (result.bboxes[0][3] - result.bboxes[0][1]) / 2 + dy)) #int(self.height / 2 + dy)

                    #int(self.height / 2 + dy)
                    # int(self.face[0][1] + result.bboxes[0][1] + (result.bboxes[0][3] - result.bboxes[0][1]) / 2 + dy) #int(self.height / 2 + dy)

                    # DADADADA
                    cv2.circle(frame, gaze_point, 25, (255, 0, 255), -1)

                    if settings.SHOW_SCREEN_GRID:
                        x_min = None
                        x_max = None
                        y_min = None
                        y_max = None
                        for i in range(settings.SCREEN_W ):
                            for j in range(settings.SCREEN_H):
                                if self.width // settings.SCREEN_W * i <= gaze_point[0] < self.width // settings.SCREEN_W * (i + 1) and self.height // settings.SCREEN_H * j <= gaze_point[1] < self.height // settings.SCREEN_H * (j + 1):
                                    x_min = self.width // settings.SCREEN_W * i
                                    x_max = self.width // settings.SCREEN_W * (i + 1)
                                    y_min = self.height // settings.SCREEN_H * j
                                    y_max = self.height // settings.SCREEN_H * (j + 1)

                                    sub_img = frame[y_min:y_max, x_min:x_max]
                                    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                                    frame[y_min:y_max, x_min:x_max] = res
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                                    self.row = i + 1

                                    self.col = j + 1

                                    self.tp += 1
                                    # cv2.putText(frame, f'{i+1, j + 1}', (self.width // 2, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 3)

                                    break
                        if not (x_min and x_max and y_min and y_max):
                            cv2.rectangle(frame, (0,0), (self.width, self.height), (255, 255, 0), 5)
                    # gaze_point = int(self.width / 2 + dx), int(self.height / 2 )
                    if len(self.points) == 3:
                        self.points.pop(0)
                    self.points.append(gaze_point)
                    print(self.points)

                    if len(self.points) == 3:
                        gaze_point_x = round(mean([self.points[0][0],self.points[1][0], self.points[2][0]]))
                        gaze_point_y = round(mean([self.points[0][1],self.points[1][1], self.points[2][1]]))
                        # cv2.circle(frame, (gaze_point_x, gaze_point_y), 25, (0, 0, 255), -1)
                    # else:

                    # test
                    DISTANCE_TO_OBJECT = self.distance * 10 # mm
                    HEIGHT_OF_HUMAN_FACE = 250
                    length_per_pixel = HEIGHT_OF_HUMAN_FACE / (result.bboxes[0][3] - result.bboxes[0][1])
                    gaze_pitch = result.yaw[0]
                    gaze_yaw = result.pitch[0]
                    # print(length_per_pixel, self.width / 2)
                    # max yaw = 30
                    # dx = -(self.width / (self.distance / 100)) * np.tan(gaze_yaw * (self.distance / 100))
                    dx = -self.width * np.tan(gaze_yaw * (self.distance / 100) + face_yaw)
                    # print(dx)
                    dx = dx if not np.isnan(dx) else 100000000
                    dx = dx * 1
                    if gaze_pitch <= face_pitch:
                        dy = -DISTANCE_TO_OBJECT * np.arccos(gaze_yaw) * np.tan(gaze_pitch) / length_per_pixel
                    else:
                        dy = -DISTANCE_TO_OBJECT * np.arccos(gaze_yaw) * np.tan(gaze_pitch - face_pitch) / length_per_pixel
                    dy = dy if not np.isnan(dy) else 100000000

                    gaze_point = int(self.width / 2 + dx), int(self.height / 2 + dy)
                    # gaze_point = int(self.width / 2 + dx), int(self.height / 2)

                    # cv2.circle(frame, gaze_point, 100, (0, 255, 255), -1)
                    if settings.SHOW_TEXT:
                        cv2.putText(frame, f'gaze yaw: {math.degrees(gaze_yaw):.0f}', (self.face[1][0], self.face[0][1] + 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                        cv2.putText(frame, f'gaze yaw: {math.degrees(gaze_yaw):.0f}', (self.face[1][0], self.face[0][1] + 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(frame, f'gaze pitch: {math.degrees(gaze_pitch):.0f}',
                                    (self.face[1][0], self.face[0][1] + 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                        cv2.putText(frame, f'gaze pitch: {math.degrees(gaze_pitch):.0f}',
                                    (self.face[1][0], self.face[0][1] + 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


                if 1 == 0:
                    cv2.line(frame, (0, frame.shape[0] // 2), (round(self.distance * 38), frame.shape[0] // 2), (255, 0, 0),
                             3)
                    # print(self.__pitch, round(result.yaw[0] / np.pi * 180))
                    pitch = self.__pitch - round(result.yaw[0] / np.pi * 180)
                    face_pitch = math.sin(self.__pitch * np.pi / 180)
                    gaze_pitch = result.yaw[0]
                    tmp = round(self.distance * 38 * (face_pitch + gaze_pitch))
                    print(tmp)
                    # self.face_frame = render(self.face_frame, result)
                    face_yaw = math.sin(self.__yaw * np.pi / 180)
                    gaze_yaw = result.pitch[0]

                    tmp_yaw = round(self.distance * 38 * (face_yaw + gaze_yaw))
                    # print(result)

                    tmp_face = round(self.distance * 30 * face_pitch)
                    cv2.circle(frame, (frame.shape[1] // 2 - tmp_yaw, frame.shape[0] // 2), 15, (255, 255, 0), -1)

                    cv2.circle(frame, (frame.shape[1] // 2, -tmp), 15, (0, 255, 0), -1)
                    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2 - tmp_face), 15, (0, 255, 255), -1)
                    cv2.putText(frame, f'yaw {gaze_yaw:.2f}',(50, 200),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                    cv2.putText(frame, f'pitch {gaze_pitch:.2f}',(50, 300),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            except ValueError :
                print('need at least one array to stack')


        # print(self.pupil_left_coords(), 1111)
        # input()
        return frame
