import cv2
import numpy as np
import mediapipe as mp

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def putText(img: np.ndarray, text: str, pos: cv2.typing.Point, color: cv2.typing.Scalar) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA, False)


def get_landmarks(img: np.ndarray):
    img_h, img_w, _ = img.shape
    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        landmark_points = face_mesh.process(img).multi_face_landmarks
        return landmark_points


def get_mesh_points(img: np.ndarray, landmarks):
    img_h, img_w, _ = img.shape
    mesh_points = np.array(
        [np.multiply([p.x, p.y], [img_w, img_w]).astype(int) for p in landmarks[0].landmark]
    )
    return mesh_points
