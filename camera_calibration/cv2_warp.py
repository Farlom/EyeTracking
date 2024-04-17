import cv2

image = cv2.imread('../tracking/photos/img.png')
warped = cv2.warpPerspective(image, 10, (1366, 768))
cv2.imshow('frame', warped)
cv2.destroyAllWindows()