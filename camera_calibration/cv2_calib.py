import cv2
import numpy as np
import cv2 as cv
import glob

CHESSBOARD_WIDTH = int(8)
CHESSBOARD_HEIGHT = int(6)
MTX = 0
DIST = 0
NEWCAMERAMTX = 0
ROI = 0
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# ( 0:chessboard_height, 0:chessboard_width, 0 )
objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# images = glob.glob('a3_chess/*.png')
images = glob.glob('../tools/images2/img5.png')

# top: ../tools/images2/img2.png
# ../tools/images2/img5.png img6 img9
# ../tools/images/img22.png 25
for fname in images:
    img = cv.imread(fname)
    cpy = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        # print(objp)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        cv.imshow('cpy', cpy)
        cv.waitKey(500)

        # print(objpoints, imgpoints)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(fname)
        img = cv.imread(fname)
        h, w = img.shape[:2]
        print(h, w)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print(type(newcameramtx), newcameramtx.shape)
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        MTX = mtx
        DIST = dist
        NEWCAMERAMTX = newcameramtx
        # crop the image
        x, y, w, h = roi
        print(roi)
        ROI = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite('calibresult.png', dst)
        # np.save('newcameramtx.npy', newcameramtx)
        # np.save('mtx.npy', mtx)
        # np.save('dist.npy', dist)
        np.save('roi.npy', roi)
cap = cv2.VideoCapture('../tracking/videos/output_120cm.avi')
print(ROI)
while True:
    newcameramtx = np.load('newcameramtx.npy')
    ret, frame = cap.read()
    frame = cv2.undistort(frame, MTX, DIST, None, NEWCAMERAMTX)
    x, y, w, h = ROI
    # x = round((x / (1800 / 100)) * (1080 / 100))
    # y = round((y / (2880 / 100)) * (1920 / 100))
    # w = round((w / (1800 / 100)) * (1080 / 100))
    # h = round((h / (2880 / 100)) * (1920 / 100))


    print(x)
    frame = frame[y:y + h, x:x + w]
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()