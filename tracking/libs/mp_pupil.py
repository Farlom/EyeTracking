import cv2

class MP_Pupil(object):
    def __init__(self, landmarks, points):
        self.points = points
        self.landmarks = landmarks
        self.x = None
        self.y = None
        self.detect_iris()

    # Функция определения координат зрачка
    def detect_iris(self):
        # Получение координат центра и радиуса описанной вокруг точек окружности
        (cX, cY), rad = cv2.minEnclosingCircle(self.landmarks[self.points])
        self.x = int(cX)
        self.y = int(cY)
