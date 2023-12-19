import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

boardWidth = 7
boardHeight = 5
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((boardHeight*boardWidth,3), np.float32)
objp[:,:2] = np.mgrid[0:boardWidth, 0:boardHeight].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
# Make a list of calibration images
images = glob.glob('./123/*.png')
print(images)
input()
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (boardWidth,boardHeight), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (boardWidth,boardHeight), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Read a test image
fname = './123/123.png'
img = cv2.imread(fname)
img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object and image points
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

print("Img_size:")
print(img_size)
print("\nCamera matrix:")
print(mtx)
print("\nDistor_coeffs:")
print(dist)
#print("\nR_vecs:")
#for rvec in rvecs:
     #print(rvec)
#print("\nT_matrix:")
#for tvec in tvecs:
     #print(tvec)
print ("\nError(RMS): ", rms)

# Re-projection Error
mean_error = 0
for i in range(len(objpoints)):
     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
     mean_error += error
print ("\nTotal error: ", mean_error/len(objpoints))

newcameramtx, roi = cv2.getOptimalNewCameraMatrix (mtx, dist, img_size, 1, img_size)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite('./frames_test/Undist.png',dst)

img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (boardWidth,boardHeight), None)
objp = objp.astype('float32')
corners = corners.astype('float32')
_, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

print("\nR_vec_test:")
print(rvec)

print("\nT_vec_test:")
print(tvec)

# Calculate euler angle
print("\nR_matrix_test:")
R_matrix, _ = cv2.Rodrigues(rvec)
print(R_matrix)
print(tvec)
Rt = np.concatenate((R_matrix, tvec), axis=1)
print("\nRt_matrix_test:")
print(Rt)

beta = asin(Rt[2,1])
beta_dgrs = 180*beta/3.14159
print ("\nbeta: ", beta_dgrs)

alpha = asin(-Rt[2,0]/cos(beta))
#alpha = acos(Rt_matrix[2,2]/cos(beta)) # acos дает только полож. результаты
alpha_dgrs = 180*alpha/3.14159
print ("alpha: ", alpha_dgrs)

gamma = asin(-Rt[0,1]/cos(beta))
gamma_dgrs = 180*gamma/3.14159
print ("gamma: ", gamma_dgrs)

n=(Rt[1,3]*Rt[0,1]-Rt[0,3]*Rt[1,1])/(Rt[1,0]*Rt[0,1]-Rt[0,0]*Rt[1,1])
print ("n: ", n)

p= (Rt[0,3]-n*Rt[0,0])/Rt[0,1]
print ("p: ", p)

h= Rt[2,3]-p*Rt[2,1]-n*Rt[2,0]
print ("h: ", h)

# Visualize calibrated result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=10)
plt.show()