from __future__ import division
import os
import cv2
import numpy as np
import mediapipe as mp
from mp_eye import Eye
from calibration import Calibration
import settings
import math


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
        self.left_eye = None
        self.right_eye = None
        self.detector = detector

        self.x_left_old = 0
        self.y_left_old = 0
        self.landmarks = None
        self.pts = []

        self.updown_anle = None
        self.__pitch = None
        self.__yaw = None
        self.__roll = None
        # _face_detector is used to detect faces
        # self._face_detector = dlib.get_frontal_face_detector()
        self.__face_detector = mp.solutions.face_detection.FaceDetection()
        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        # self._predictor = dlib.shape_predictor(model_path)
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
            epsilon = 10
            result = result.detections[0].location_data.relative_bounding_box
            x1 = round(result.xmin * width)

            if self.face and abs(self.face[0][0] - x1) <= epsilon:
                x1 = self.face[0][0]

            y1 = round(result.ymin * height)

            if self.face and abs(self.face[0][1] - y1) <= epsilon:
                y1 = self.face[0][1]

            x2 = round((result.xmin + result.width) * width)

            if self.face and abs(self.face[1][0] - x2) <= epsilon:
                x2 = self.face[1][0]

            y2 = round((result.ymin + result.height) * height)

            if self.face and abs(self.face[1][1] - y2) <= epsilon:
                y2 = self.face[1][1]
            # return [(x1, y1), (x2, y2)]
            faces = [(x1, y1), (x2, y2)]
            self.face = faces
            # print(faces)
            face_frame = frame[y1:y2, x1:x2]
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
            # try:
            #     # landmarks = self._predictor(frame, faces[0])
            #     landmarks = self.__face_landmarker.process(face_frame).multi_face_landmarks[0].landmark
            #     landmarks = np.array(
            #     [np.multiply([p.x, p.y], [face_frame.shape[1], face_frame.shape[0]]).astype(int) for p in landmarks]
            # )
            #     # print(landmarks[10].x)
            #     self.eye_left = Eye(face_frame, landmarks, 0, self.calibration)
            #     self.eye_right = Eye(face_frame, landmarks, 1, self.calibration)
            # except IndexError:
            #
            #     self.eye_left = None
            #     self.eye_right = None
    #     with self.__face_detector as detector:
    #         result = detector.process(frame)
    #         # cv2.imwrite('frame1.png', frame)
    #         input()
    #         print(result.detections)
    #         # if not result.detections:
    #         #     return None
    #     if result.detections:
    #         result = result.detections[0].location_data.relative_bounding_box
    #         x1 = round(result.xmin * width)
    #         y1 = round(result.ymin * height)
    #         x2 = round((result.xmin + result.width) * width)
    #         y2 = round((result.ymin + result.height) * height)
    #         # return [(x1, y1), (x2, y2)]
    #         faces = [(x1, y1), (x2, y2)]
    #         self.face = faces
    #         print(faces)
    #         face_frame = frame[y1:y2, x1:x2]
    #
    #         """Detects the face and initialize Eye objects"""
    # # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    # # faces = self._face_detector(frame)
    #         try:
    #             # landmarks = self._predictor(frame, faces[0])
    #             landmarks = self.__face_landmarker.process(face_frame).multi_face_landmarks[0].landmark
    #             landmarks = np.array(
    #             [np.multiply([p.x, p.y], [face_frame.shape[1], face_frame.shape[0]]).astype(int) for p in landmarks]
    #         )
    #             # print(landmarks[10].x)
    #             self.eye_left = Eye(face_frame, landmarks, 0, self.calibration)
    #             self.eye_right = Eye(face_frame, landmarks, 1, self.calibration)
    #
    #         except IndexError:
    #             self.eye_left = None
    #             self.eye_right = None
    #         except UnboundLocalError:
    #             self.eye_left = None
    #             self.eye_right = None
            # else:
            #     self.eye_left = None
            #     self.eye_right = None

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

        cv2.line(frame, (int((A[0] + B[0]) / 2),int((A[1] + B[1]) / 2)), (int((A[0] + B[0]) / 2),int(((A[1] + B[1]) / 2) + d)), (255,255,255, 3))

        alpha = ((B[0] - A[0]) * (E[1] - A[1])) - ((E[0] - A[0]) * (B[1] - A[1]))
        print(alpha)
        alpha = d * ((A[0] - B[0])**2 + (A[1] - B[1])**2)
        print(alpha)
        alpha = ((B[0]-A[0]) * (E[1]-A[1]) - (E[0] - A[0]) * (B[1] - A[1])) / (d * ((A[0] - B[0])**2 + (A[1] - B[1]) **2))
        alpha = math.acos(alpha)
        alpha = (alpha * 180) / math.pi
        # print(alpha, 11111111111111111)
        # cv2.putText(frame, f'{alpha}', (450, frame.shape[0] - 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.circle(frame, A_frame, 3, (255, 255, 255), 1)
        cv2.circle(frame, B_frame, 3, (255, 255, 255), 1)
        cv2.circle(frame, C_frame, 3, (255, 255, 255), 1)
        cv2.circle(frame, D_frame, 3, (255, 255, 255), 1)
        cv2.circle(frame, E_frame, 3, (255, 255, 255), 1)




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

        cv2.line(img,
                 (x_left_circle, y_left_circle),
                 (
                     int(x_left_circle - (rad_left_circle // 2) * math.cos(self.__yaw)),
                     int(y_left_circle - (rad_left_circle // 2) * math.sin(self.__pitch))
                 ), (255, 0, 255), 3)

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

        try:
            alpha = (p2[1] - p1[1])/(p2[0] - p1[0])
            angle1 = str(-1 * int(math.degrees(math.atan(alpha))))
            self.updown_anle = angle1
        except ZeroDivisionError:
            angle1 = '90'
            self.updown_anle = angle1
        cv2.putText(frame, f'Up/down: {angle1}', (450, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

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

        cv2.putText(frame, f'pitch (up down): {pitch}', (450, frame.shape[0] - 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.putText(frame, f'roll (tilt): {roll}', (450, frame.shape[0] - 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.putText(frame, f'yaw (left right): {yaw}', (450, frame.shape[0] - 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

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



        (point_2d, _) = cv2.projectPoints((
            (
                float((self.pupil_left_coords()[0] + self.face[0][0]) * (front_size / front_size)),
                float((self.pupil_left_coords()[1] + self.face[0][1]) * (front_size / rear_size)),
                      front_depth)
        ),
          rotation_vector,
          translation_vector,
          camera_matrix,
          dist_coeffs)
        # cv2.circle(frame, (round(point_2d[0][0][0]), round(point_2d[0][0][1])), 2, (0, 0, 255), 5)

        (point_2d, _) = cv2.projectPoints((
            (
                float(self.pupil_right_coords()[0] * (front_size / front_size)),
                float(self.pupil_right_coords()[1] * (front_size / rear_size)),
                front_depth)
        ),
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs)
        # cv2.circle(frame, (round(point_2d[0][0][0]), round(point_2d[0][0][1])), 2, (0, 0, 255), 5)
        # print(point_2d, round(point_2d[0][0][0]), round(point_2d[0][0][1]))
        # # test
        # image_points = np.array([
        #     NOSE_TIP,
        #     CHIN,
        #     LEFT_EYE_CORNER,
        #     RIGHT_EYE_CORNER,
        #     LEFT_MOUTH_CORNER,
        #     RIGHT_MOUSE_CORNER,
        #     (self.pupil_left_coords()[0] + self.face[0][0], self.pupil_left_coords()[1] + self.face[0][1]),
        #     (self.pupil_right_coords()[0] + self.face[0][0], self.pupil_right_coords()[1] + self.face[0][1])
        # ], dtype='double')
        #
        # model_points = np.array([
        #     (0.0, 0.0, 0.0),  # Nose tip
        #     (0.0, -330.0, -65.0),  # Chin
        #     (-225.0, 170.0, -135.0),  # Left eye left corner
        #     (225.0, 170.0, -135.0),  # Right eye right corne
        #     (-150.0, -150.0, -125.0),  # Left Mouth corner
        #     (150.0, -150.0, -125.0),  # Right mouth corner
        #     (-135.0, 170.0, -135.0),
        #     (135.0, 170.0, -135.0)
        #
        # ])
        #
        # focal_length = frame.shape[1]
        # center = (frame.shape[1] / 2, frame.shape[0] / 2)
        # camera_matrix = np.array(
        #     [[focal_length, 0, center[0]],
        #      [0, focal_length, center[1]],
        #      [0, 0, 1]], dtype="double"
        # )
        #
        # # print("Camera Matrix :\n {0}".format(camera_matrix))
        #
        # dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
        #                                                               dist_coeffs)
        # if output:
        #     print("Camera Matrix :\n {0}".format(camera_matrix))
        #     print("Rotation Vector:\n {0}".format(rotation_vector))
        #     print("Translation Vector:\n {0}".format(translation_vector))
        #
        # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(-135.0, 170.0, 1000.0)]), rotation_vector,
        #                                                  translation_vector,
        #                                                  camera_matrix, dist_coeffs)
        #
        # for p in image_points:
        #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        #
        # p1 = (int(image_points[6][0]), int(image_points[6][1]))
        # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        # try:
        #     alpha = (p2[1] - p1[1]) / (p2[0] - p1[0])
        #     angle1 = str(-1 * int(math.degrees(math.atan(alpha))))
        # except:
        #     angle1 = 90
        # cv2.putText(frame, f'Up/down: {angle1}', (450, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        #
        # cv2.line(frame, p1, p2, (255, 255, 0), 2)
        #
        # # endtest

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()
        epsilon = 0
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left, x_left_center, y_left_center = self.pupil_left_coords()
            # if abs(x_left_center - self.x_left_old) > epsilon and abs(y_left_center - self.y_left_old) > epsilon:
            #     self.x_left_old = x_left_center
            #     self.y_left_old = y_left_center
            # else:
            #     x_left_center = self.x_left_old
            #     y_left_center = self.y_left_old

            x_left += self.face[0][0]
            y_left += self.face[0][1]
            x_left_center += self.face[0][0]
            y_left_center += self.face[0][1]
            self.left_eye = (x_left, y_left)

            x_right, y_right, x_right_center, y_right_center = self.pupil_right_coords()
            x_right += self.face[0][0]
            y_right += self.face[0][1]
            x_right_center += self.face[0][0]
            y_right_center += self.face[0][1]

            # tilt
            delta_x = x_left_center - x_right_center
            delta_y = y_left_center - y_right_center
            angle = math.atan(delta_y / delta_x)
            angle = str(int((angle * 180) / math.pi))
            cv2.putText(frame, f'tilt: {angle}', (300, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)

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
            # cv2.circle(frame, (x_right_center, y_right_center + int(-10 * math.sin(int(self.updown_anle)))), 1, (255,0,255), 2)
            # cv2.circle(frame, (x_left_center, y_left_center + int(-10 * math.cos(int(self.updown_anle)))), 1, (255,0,255), 2)


            # new ptrs
            self.__sphere(frame)

            # self.pts.append((x_left, y_left))
            # if len(self.pts) > 100:
            #     self.pts.pop(0)
            # for point in self.pts:
            #     cv2.circle(frame, point, 1, color, 1)

            # face rect
            cv2.rectangle(frame, (self.face[0][0], self.face[0][1]), (self.face[1][0], self.face[1][1]), color)
            cv2.line(frame, (x_left_center, y_left_center), (x_left, y_left), color)

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

            # if x_left_center > x_left:
            #     cv2.line(frame, (x_left_center, y_left_center), (0, eye_left_b), color)
            # else:
            #     cv2.line(frame, (x_left_center, y_left_center), (frame.shape[1], round(frame.shape[1] * eye_left_k + eye_left_b)), color)

            # if x_right_center > x_right:
            #     cv2.line(frame, (x_right_center, y_right_center), (0, eye_right_b), color)
            # else:
            #     cv2.line(frame, (x_right_center, y_right_center),
            #              (frame.shape[1], round(frame.shape[1] * eye_right_k + eye_right_b)), color)

            # tmp_k_r = (r_center_y - cY_r) / (r_center_x - cX_r)
            # tmp_b_r = cY_r - (tmp_k_r * cX_r)
            x1 = 0
            y1 = 0
            # for i in range(3):
            #     if i == 0:
            #         x1 = 50
            #     elif i == 1:
            #         x1 = frame.shape[1] // 2
            #     else:
            #         x1 = frame.shape[1] - 50
            #
            #     for j in range(3):
            #         if j == 0:
            #             y1 = 50
            #         elif j == 1:
            #             y1 = frame.shape[0] // 2
            #         else:
            #             y1 = frame.shape[0] - 50



            cv2.circle(frame, (50 ,50), 5, (255,255,255), -1)
            cv2.circle(frame, (frame.shape[1] //2,50), 5, (255,255,255), -1)
            cv2.circle(frame, (frame.shape[1] - 50 ,50), 5, (255,255,255), -1)

            cv2.circle(frame, (50, frame.shape[0] // 2), 5, (255, 255, 255), -1)
            cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (255, 255, 255), -1)
            cv2.circle(frame, (frame.shape[1] - 50, frame.shape[0] // 2), 5, (255, 255, 255), -1)

            cv2.circle(frame, (50, frame.shape[0] - 50), 5, (255, 255, 255), -1)
            cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] - 50), 5, (255, 255, 255), -1)
            cv2.circle(frame, (frame.shape[1] - 50, frame.shape[0] - 50), 5, (255, 255, 255), -1)


        return frame
