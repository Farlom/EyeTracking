import numpy as np
import cv2 as cv
import glob

CHESSBOARD_WIDTH = int(6)
CHESSBOARD_HEIGHT = int(6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# ( 0:chessboard_height, 0:chessboard_width, 0 )
objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/calib_radial.jpg')
for fname in images:
    img = cv.imread(fname)
    cpy = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(corners[0])

        for i in enumerate(corners):
            print(i[1])
        for corner in corners:
            for coord in corner:
                cv.line(cpy, (int(coord[0]), int(coord[1])), (int(corners[1][0][0]), int(corners[1][0][1])), (0, 0, 255), 2, cv.LINE_AA)
        # print(objp)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), corners2, ret)
        cv.imshow('img', img)
        cv.imshow('cpy', cpy)
        cv.waitKey(50000)

        # print(objpoints, imgpoints)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        img = cv.imread('images/calib_radial.jpg')
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite('calibresult.png', dst)
cv.destroyAllWindows()