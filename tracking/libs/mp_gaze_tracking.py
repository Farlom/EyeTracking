from __future__ import division
import os
import cv2
import numpy as np
import mediapipe as mp
from mp_eye import Eye
from calibration import Calibration
import settings
import math
from statistics import mean
# from gaze_estimation import GazeEstimation

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    EPSILLON = 5
    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self, detector):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.face = None
        self.face_frame = None
        self.left_eye = None
        self.right_eye = None
        self.detector = detector

        self.x_left_old = 0
        self.y_left_old = 0
        self.landmarks = None
        self.pts = []

        self.updown_anle = None
        self.__pitch = None
        self.yaw = None
        self.__roll = None
        self.vector = None
        # _face_detector is used to detect faces
        self.__face_detector = mp.solutions.face_detection.FaceDetection()

        # _predictor is used to get facial landmarks of a given face
        self.__face_landmarker = mp.solutions.face_mesh.FaceMesh(refine_landmarks=False)
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
        height, width = frame.shape[:2]
        result = self.detector.process(frame)
        # cv2.imwrite('frame1.png', frame)

        if result.detections:
            result = result.detections[0].location_data.relative_bounding_box

            epsilon = 10

            x1 = round(result.xmin * width)

            y1 = round(result.ymin * height)

            x2 = round((result.xmin + result.width) * width)

            y2 = round((result.ymin + result.height) * height)

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
            face_frame = frame[y1:y2, x1:x2]
            self.face_frame = face_frame
            # cv2.imshow('f',face_frame)
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.9
            ) as mesh:
                res_landmarks = mesh.process(face_frame).multi_face_landmarks
                # print(res_landmarks)
                if res_landmarks:
                    landmarks = res_landmarks[0].landmark
                    landmarks = np.array(
                        [np.multiply([p.x, p.y], [face_frame.shape[1], face_frame.shape[0]]).astype(int) for p in
                         landmarks]
                    )

                    if type(self.landmarks) != type(None):
                        for i in range(len(landmarks)):
                            if abs(landmarks[i][0] - self.landmarks[i][0]) >= self.EPSILLON //2 or abs(
                                    landmarks[i][1] - self.landmarks[i][1]) >= self.EPSILLON // 2:
                                self.landmarks = landmarks
                            else:
                                landmarks = self.landmarks
                    self.landmarks = landmarks
                    self.eye_left = Eye(face_frame, landmarks, 0, self.calibration)
                    self.eye_right = Eye(face_frame, landmarks, 1, self.calibration)

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            cx = self.eye_left.origin[0] + round(self.eye_left.center[0])
            cy = self.eye_left.origin[1] + round(self.eye_left.center[1])
            return x, y, cx, cy

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            cx = self.eye_right.origin[0] + round(self.eye_right.center[0])
            cy = self.eye_right.origin[1] + round(self.eye_right.center[1])
            return x, y, cx, cy

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def __draw_rays(self):
        ...

    def __normalize_frame(self, frame):
        ...
    def __face_rotation(self, frame):
        # print(self.landmarks[234], 123)
        # a = (self.face[0][0] + self.landmarks[143][0], self.face[0][1] + self.landmarks[143][1])
        # f = (self.face[0][0] + self.landmarks[372][0], self.face[0][1] + self.landmarks[372][1])
        # print(self.face[0][0] + self.landmarks[234][0], 11111111)
        # cv2.circle(frame, self.face[0], 3, (255,255,255),5)
        # cv2.circle(frame, a, 3, (255, 0, 255), 5)
        # cv2.circle(frame, f, 3, (255, 0, 255), 5)

        # http://www.nauteh-journal.ru/files/fb62092c-a158-4e60-a685-538caed492f2

        # print(self.pupil_left_coords(), 1111)
        A_frame = (self.face[0][0] + self.pupil_left_coords()[0], self.face[0][1] + self.pupil_left_coords()[1])
        B_frame = (self.face[0][0] + self.pupil_right_coords()[0], self.face[0][1] + self.pupil_right_coords()[1])
        C_frame = (self.face[0][0] + self.landmarks[4][0], self.face[0][1] + self.landmarks[4][1])
        D_frame = (self.face[0][0] + self.landmarks[14][0], self.face[0][1] + self.landmarks[14][1])
        E_frame = (self.face[0][0] + self.landmarks[152][0], self.face[0][1] + self.landmarks[152][1])

        A = (self.pupil_left_coords()[0], self.pupil_left_coords()[1])
        B = (self.pupil_right_coords()[0], self.pupil_right_coords()[1])
        C = (self.landmarks[4][0], self.landmarks[4][1])
        D = (self.landmarks[14][0], self.landmarks[14][1])
        E = (self.landmarks[152][0], self.landmarks[152][1])

        d = E[1] - ((A[1] + B[1]) / 2)

        # cv2.line(frame, (int((A[0] + B[0]) / 2),int((A[1] + B[1]) / 2)), (int((A[0] + B[0]) / 2),int(((A[1] + B[1]) / 2) + d)), (255,255,255, 3))

        alpha = ((B[0] - A[0]) * (E[1] - A[1])) - ((E[0] - A[0]) * (B[1] - A[1]))
        # print(alpha)
        alpha = d * ((A[0] - B[0])**2 + (A[1] - B[1])**2)
        # print(alpha)
        alpha = ((B[0]-A[0]) * (E[1]-A[1]) - (E[0] - A[0]) * (B[1] - A[1])) / (d * ((A[0] - B[0])**2 + (A[1] - B[1]) **2))
        alpha = math.acos(alpha)
        alpha = (alpha * 180) / math.pi
        # print(alpha, 11111111111111111)
        # cv2.putText(frame, f'{alpha}', (450, frame.shape[0] - 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        # cv2.circle(frame, A_frame, 3, (255, 255, 255), 1)
        # cv2.circle(frame, B_frame, 3, (255, 255, 255), 1)
        # cv2.circle(frame, C_frame, 3, (255, 255, 255), 1)
        # cv2.circle(frame, D_frame, 3, (255, 255, 255), 1)
        # cv2.circle(frame, E_frame, 3, (255, 255, 255), 1)

        # pts1 = np.float32([[A_frame[0], A_frame[1]],
        #            [B_frame[0], B_frame[1]],
        #            [E_frame[0], E_frame[1]]])
        # pts2 = np.float32([[0, 0],
        #            [0, 200],
        #            [200, 100]])
        # M = cv2.getAffineTransform(pts1, pts2)
        # # M[0][2] = 0
        # # M[1][2] = 0
        # dst = cv2.warpAffine(frame, M, (frame.shape[0], frame.shape[1]))
        # dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        # print('MMMMMMMMMM', M)
        # # cv2.imshow('dst', dst)
        # center = (dst.shape[1] // 2, dst.shape[0] // 2)
        # angle = 0
        # scale = 0.9
        # rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        # warp_rotate_dst = cv2.warpAffine(dst, rot_mat, (dst.shape[1], dst.shape[0]))
        # cv2.imshow('dst', warp_rotate_dst)


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

    def __sphere(self, img):
        x_left, y_left, x_left_center, y_left_center = self.pupil_left_coords()
        x_left += self.face[0][0]
        y_left += self.face[0][1]

        x_right, y_right, x_right_center, y_right_center = self.pupil_right_coords()
        x_right += self.face[0][0]
        y_right += self.face[0][1]

        color = (0, 255, 0)

        (x_left_circle, y_left_circle), rad_left_circle = cv2.minEnclosingCircle(
            self.landmarks[self.LEFT_EYE_POINTS])
        x_left_circle = int(x_left_circle + self.face[0][0])
        y_left_circle = int(y_left_circle + self.face[0][1])

        (x_right_circle, y_right_circle), rad_right_circle = cv2.minEnclosingCircle(
            self.landmarks[self.RIGHT_EYE_POINTS])
        x_right_circle = int(x_right_circle + self.face[0][0])
        y_right_circle = int(y_right_circle + self.face[0][1])

        # cv2.circle(img, (x_left_circle, y_left_circle), int(rad_left_circle), (255, 255, 255), 2)
        cv2.line(img, (x_left_circle - 5, y_left_circle), (x_left_circle + 5, y_left_circle), (255, 255, 255), 1)
        cv2.line(img, (x_left_circle, y_left_circle), (x_left_circle, y_left_circle + 5), (255, 255, 255), 1)

        # cv2.circle(img, (x_right_circle, y_right_circle), int(rad_right_circle), (255, 255, 255), 2)
        cv2.line(img, (x_right_circle - 5, y_right_circle), (x_right_circle + 5, y_right_circle), (255, 255, 255), 1)
        cv2.line(img, (x_right_circle, y_right_circle), (x_right_circle, y_right_circle + 5), (255, 255, 255), 1)

        # cv2.line(img,
        #          (x_left_circle, y_left_circle),
        #          (
        #              int(x_left_circle - 15),
        #              int(y_left_circle - 15 * math.sin(self.__pitch))
        #          ), (255, 0, 255), 3)

        # x_left_circle = int(x_left_circle - (rad_left_circle // 4) * math.cos(self.__yaw))
        # y_left_circle = int(y_left_circle - (rad_left_circle // 4) * math.sin(self.__pitch))
        # cv2.circle(img, (x_left_circle, y_left_circle), 1, color, 2)
        if x_left_circle != x_left and y_left_circle != y_left:
            eye_left_k = (y_left_circle - y_left) / (x_left_circle - x_left)
            eye_left_b = round(y_left - (eye_left_k * x_left_circle))
        else:
            eye_left_b = 0
            eye_left_k = 0

        if x_right_circle != x_right and y_right_circle != y_right:
            eye_right_k = (y_right_circle - y_right) / (x_right_circle - x_right)
            eye_right_b = round(y_right - (eye_right_k * x_right_circle))
        else:
            eye_right_b = 0
            eye_right_k = 0

        if settings.SHOW_LEFT_GAZE_RAY:
            if x_left_circle > x_left:
                cv2.line(img, (x_left_circle, y_left_circle), (0, eye_left_b), color)
            else:
                cv2.line(img, (x_left_circle, y_left_circle),
                         (img.shape[1], round(img.shape[1] * eye_left_k + eye_left_b)), color)

        if settings.SHOW_RIGHT_GAZE_RAY:
            if x_right_circle > x_right:
                cv2.line(img, (x_right_circle, y_right_circle), (0, eye_right_b), color)
            else:
                cv2.line(img, (x_right_circle, y_right_circle),
                         (img.shape[1], round(img.shape[1] * eye_right_k + eye_right_b)), color)


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

            x_left += self.face[0][0]
            y_left += self.face[0][1]
            pupil = (x_left, y_left)
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
            x_right += self.face[0][0]
            y_right += self.face[0][1]

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
                cv2.line(img, p1, p2, (0, 0, 255), 2)

                return (p1, p2)
        else:
            # 2d pupil location
            x_right, y_right, _, _ = self.pupil_right_coords()
            x_right += self.face[0][0]
            y_right += self.face[0][1]

            x_left, y_left, _, _ = self.pupil_left_coords()

            x_left += self.face[0][0]
            y_left += self.face[0][1]

            pupil = (round(mean([x_right, x_left])), round(mean([y_right, y_left])))

            # Transformation between image point to world point
            _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

            if transformation is not None:  # if estimateAffine3D secsseded
                # project pupil image point into 3d world point
                pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T

                # 3D gaze point (10 is arbitrary value denoting gaze distance)
                S = EYE_BALL_CENTER_MEAN + (pupil_world_cord - EYE_BALL_CENTER_MEAN) * 10

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

                return (p1, p2)

    def __estimate(self, frame: np.ndarray, output: bool):
        # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        # estimation
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

        # *Центр левого глаза: (-135, 170, -135)
        # *Центр правого глаза: (135, 170, -135

        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # print("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs)
        if output:
            print("Camera Matrix :\n {0}".format(camera_matrix))
            print("Rotation Vector:\n {0}".format(rotation_vector))
            print("Translation Vector:\n {0}".format(translation_vector))

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0,400.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        # for p in image_points:
        #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        (x_left_circle, y_left_circle), rad_left_circle = cv2.minEnclosingCircle(
            self.landmarks[self.LEFT_EYE_POINTS])
        x_left_circle = int(x_left_circle + self.face[0][0])
        y_left_circle = int(y_left_circle + self.face[0][1])
        (left_end_point2D, jacobian) = cv2.projectPoints(np.array([(-135.0, 170.0, -400.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        p11 = (x_left_circle, y_left_circle)
        p12 = (int(left_end_point2D[0][0][0]), int(left_end_point2D[0][0][1]))
        # cv2.line(frame, p11, p12, (0, 255, 0), 2)


        (left_end_point2D, jacobian) = cv2.projectPoints(np.array([(-135.0, 170.0, 400.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        p11 = (x_left_circle, y_left_circle)
        p12 = (int(left_end_point2D[0][0][0]), int(left_end_point2D[0][0][1]))
        if settings.SHOW_LEFT_GAZE_RAY:
            cv2.line(frame, p11, p12, (255, 0, 0), 2)

        # *Центр левого глаза: (-135, 170, -135)
        # *Центр правого глаза: (135, 170, -135
        (x_right_circle, y_right_circle), rad_right_circle = cv2.minEnclosingCircle(
            self.landmarks[self.RIGHT_EYE_POINTS])
        x_right_circle = int(x_right_circle + self.face[0][0])
        y_right_circle = int(y_right_circle + self.face[0][1])
        # print(-135.0 - rad_right_circle, 11111111111)
        (right_end_point2D, jacobian) = cv2.projectPoints(np.array([(135.0, 170.0, -200.0)]), rotation_vector,
                                                          translation_vector,
                                                          camera_matrix, dist_coeffs)

        p21 = (x_right_circle, y_right_circle)
        p22 = (int(right_end_point2D[0][0][0]), int(right_end_point2D[0][0][1]))
        # cv2.line(frame, p21, p22, (255, 255, 255), 3)
        # y1 - y2 ) / ( x1 - x2)
        # y2 - (k * x2)
        try:
            k = (p11[1] - p12[1]) / (p11[0] - p12[0])
        except ZeroDivisionError:
            k = 0

        b = p12[1] - k * p12[0]


        if p12[0] > p11[0] and p12[1] < p11[1]:
            p13 = (int(p11[0] - rad_left_circle // 2), int(k * (p11[0] - rad_left_circle // 2) + b))
            # cv2.line(frame, p11, p13, (255, 255, 255), 2)
        elif p12[0] > p11[0] and p12[1] > p11[1]:
            p13 = (int(p11[0] - rad_left_circle // 2), int(k * (p11[0] - rad_left_circle // 2) + b))
            # cv2.line(frame, p11, p13, (255, 255, 0), 2)
        elif p12[0] < p11[0] and p12[1] < p11[1]:
            p13 = (int(p11[0] + rad_left_circle // 2), int(k * (p11[0] + rad_left_circle // 2) + b))
            # cv2.line(frame, p11, p13, (255, 0, 0), 2)
        elif p12[0] < p11[0] and p12[1] > p11[1]:
            p13 = (int(p11[0] + rad_left_circle // 2), int(k * (p11[0] + rad_left_circle // 2) + b))

            # cv2.line(frame, p11, p13, (255, 0, 255), 2)
        # elif p13[0] > p11[0] and p13[1] < p11[1]:
            # p13 = (int(p11[0] + rad_left_circle // 2), int(k * (p11[0] + rad_left_circle // 2) + b))
            # cv2.line(frame, p11, p12, (255, 255, 255), 2)

        try:
            alpha = (p2[1] - p1[1])/(p2[0] - p1[0])
            angle1 = str(-1 * int(math.degrees(math.atan(alpha))))
            self.updown_anle = angle1
        except ZeroDivisionError:
            angle1 = '90'
            self.updown_anle = angle1
        # cv2.putText(frame, f'Up/down: {angle1}', (450, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        if settings.SHOW_NOSE_RAY:
            cv2.line(frame, p1, p2, (255, 0, 0), 2)



        color = (255, 255, 0)
        line_width = 1

        rear_size = 300
        rear_depth = 0
        front_size = 500
        front_depth = 400
        point_2d = self.get_2d_points(frame, rotation_vector, translation_vector, camera_matrix, [rear_size, rear_depth, front_size, front_depth])

        #https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV

        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = int(math.degrees(math.asin(math.sin(pitch))))
        self.__pitch = pitch

        roll = int(-math.degrees(math.asin(math.sin(roll))))
        self.__roll = roll

        yaw = int(math.degrees(math.asin(math.sin(yaw))))
        self.__yaw = yaw
        # cv2.line(frame, p11, (int(p11[0] + 50 * math.cos(self.__yaw)), p11[1]), (255, 255, 255), 2)
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
            cv2.line(frame, point_2d[5], point_2d[6], (255, 255, 255), 3)
            cv2.line(frame, point_2d[6], point_2d[7], (255, 255, 255), 3)
            cv2.line(frame, point_2d[7], point_2d[8], (255, 255, 255), 3)
            cv2.line(frame, point_2d[8], point_2d[9], (255, 255, 255), 3)


    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()
        epsilon = 0
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left, x_left_center, y_left_center = self.pupil_left_coords()
            x_left += self.face[0][0]
            y_left += self.face[0][1]
            x_left_center += self.face[0][0]
            y_left_center += self.face[0][1]
            l_max_x = (max(self.landmarks[self.LEFT_EYE_POINTS], key=lambda item: item[0]))[0]+ self.face[0][0]
            l_min_x = (min(self.landmarks[self.LEFT_EYE_POINTS], key=lambda item: item[0]))[0] + self.face[0][0]
            l_max_y = (max(self.landmarks[self.LEFT_EYE_POINTS], key=lambda item: item[1]))[1]+ self.face[0][1]
            l_min_y = (min(self.landmarks[self.LEFT_EYE_POINTS], key=lambda item: item[1]))[1] + self.face[0][1]
            l_width = l_max_x - l_min_x
            l_height = l_max_y - l_min_y
            self.left_eye = (x_left, y_left)

            x_right, y_right, x_right_center, y_right_center = self.pupil_right_coords()
            x_right += self.face[0][0]
            y_right += self.face[0][1]
            x_right_center += self.face[0][0]
            y_right_center += self.face[0][1]
            r_max_x = (max(self.landmarks[self.RIGHT_EYE_POINTS], key=lambda item: item[0]))[0] + self.face[0][0]
            r_min_x = (min(self.landmarks[self.RIGHT_EYE_POINTS], key=lambda item: item[0]))[0] + self.face[0][0]
            r_max_y = (max(self.landmarks[self.RIGHT_EYE_POINTS], key=lambda item: item[1]))[1] + self.face[0][1]
            r_min_y = (min(self.landmarks[self.RIGHT_EYE_POINTS], key=lambda item: item[1]))[1] + self.face[0][1]
            r_width = r_max_x - r_min_x
            r_height = r_max_y - r_min_y


            # tilt
            delta_x = x_left_center - x_right_center
            delta_y = y_left_center - y_right_center
            angle = math.atan(delta_y / delta_x)
            angle = str(int((angle * 180) / math.pi))
            # cv2.putText(frame, f'tilt: {angle}', (300, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)

            # rotate
            self.__face_rotation(frame)
            # ab = self.eye_left.origin[0]
            # face_width = self.face[1][0] - self.face[0][0]
            # ef = face_width - (2 * self.pupil_right_coords()[2]- self.eye_right.origin[0])
            # c = ab / ef
            # print(ab, ef, face_width, c)
            # cv2.circle(frame,
            #            (ab + self.face[0][0], self.eye_left.origin[1] + self.face[0][1]),
            #            1, color, 2)

            # cv2.circle(frame,
            #            (ef + self.face[0][0], self.eye_right.origin[1] + self.face[0][1]),
            #            1, color, 2)

            # estimation
            self.__estimate(frame, False)


            # center ptr
            # cv2.circle(frame,
            #            (x_left_center, y_left_center),
            #            1, color, 2)

            # ptrs
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)



            # new ptrs
            # self.__sphere(frame)

            # self.pts.append((x_left, y_left))
            # if len(self.pts) > 100:
            #     self.pts.pop(0)
            # for point in self.pts:
            #     cv2.circle(frame, point, 1, color, 1)

            # face rect
            cv2.rectangle(frame, (self.face[0][0], self.face[0][1]), (self.face[1][0], self.face[1][1]), color)
            # cv2.line(frame, (x_left_center, y_left_center), (x_left, y_left), color)

            # frame = self.frame.copy()
            if x_left_center != x_left and y_left_center != y_left:
                eye_left_k = (y_left_center - y_left) / (x_left_center - x_left)
                eye_left_b = round(y_left - (eye_left_k * x_left_center))
            else:
                eye_left_b = 0
                eye_left_k = 0

            if x_right_center != x_right and y_right_center != y_right:
                eye_right_k = (y_right_center - y_right) / (x_right_center - x_right)
                eye_right_b = round(y_right - (eye_right_k * x_right_center))
            else:
                eye_right_b = 0
                eye_right_k = 0

            # self.__3d_to_2d(frame, 0)
            # self.__3d_to_2d(frame, 1)
            self.vector = self.__3d_to_2d(frame, 2)[1]


            # cv2.circle(frame, (frame.shape[1] //2, frame.shape[0] // 2), 3, (255, 255, 255), -1)
            # cv2.rectangle(frame, (100, 100), (frame.shape[1] - 100, frame.shape[0] - 100), (255, 0,0), 1)
        return frame
