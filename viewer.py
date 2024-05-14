import cv2

cap = cv2.VideoCapture('tracking/videos/output_120cm.avi')
mtx = open('camera_calibration/mtx.txt', 'r').read()
dist = open('camera_calibration/dist.txt', 'r').read()
newcameramtx = open('camera_calibration/newcameramtx', 'r').read()
while True:
    ret, frame = cap.read()
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()