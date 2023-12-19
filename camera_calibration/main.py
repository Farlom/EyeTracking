# This is a sample Python script.
import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

cap = cv2.VideoCapture("rtsp://admin:vide0-II@172.20.6.234:554")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920,  1080))


while True:
    ret, frame = cap.read()
    cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_AREA)

    cv2.imshow('frame', frame)
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()