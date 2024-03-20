import math
import numpy as np
import cv2
from mp_pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        cv2.imwrite('face_frame.png', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        region = np.array([landmarks[p] for p in points], dtype=np.int32)
        # print(region)
        self.landmark_points = region

        # Applying a mask to get only the eye
        invert_frame = cv2.bitwise_not(frame)
        invert_frame = cv2.cvtColor(invert_frame, cv2.COLOR_BGR2GRAY)
        width, height = invert_frame.shape[:2]


        mask = np.zeros((width, height), dtype=np.uint8)
        cv2.fillPoly(mask, [region], (255, 255, 255))
        # print(mask.shape, invert_frame.shape)
        eye = cv2.bitwise_and(invert_frame, invert_frame, mask=mask)
        # eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        eye = cv2.bitwise_not(eye)

        # Cropping on the eye
        margin = 5
        max_x = (max(region, key=lambda item: item[0]))[0] + margin
        min_x = (min(region, key=lambda item: item[0]))[0] - margin
        max_y = (max(region, key=lambda item: item[1]))[1] + margin
        min_y = (min(region, key=lambda item: item[1]))[1] - margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        # print(self.frame.shape)
        # cv2.imwrite('eye_frame.png', self.frame)
        # cv2.imshow('eyeye', self.frame)

        self.origin = (min_x, min_y)
        cv2.imshow('s', self.frame)
        height, width = self.frame.shape[:2]
        width -= 2 * margin * 0
        height -= 2 * margin * 0
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        # left = (landmarks[234].x * self.frame.shape[1], landmarks[234].y * self.frame.shape[0])
        # right = (landmarks[454].x * self.frame.shape[1], landmarks[454].y * self.frame.shape[0])
        # top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        # bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))
        #
        # eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        # eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))
        #
        # try:
        #     ratio = eye_width / eye_height
        # except ZeroDivisionError:
        #     ratio = None
        ratio = 1
        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
