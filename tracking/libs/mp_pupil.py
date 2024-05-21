import numpy as np
import cv2


class MP_Pupil(object):
    def __init__(self, landmarks, points):
        self.points = points
        self.landmarks = landmarks
        self.x = None
        self.y = None
        self.detect_iris()

    def detect_iris(self):
        (cX, cY), rad = cv2.minEnclosingCircle(self.landmarks[self.points])
        self.x = int(cX)
        self.y = int(cY)
        print(self.x)
