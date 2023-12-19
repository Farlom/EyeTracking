import cv2 # OpenCV library
import lensfunpy
focal_length = 2.0
aperture = 2.0
distance = 2
image_path = '1.png'
undistorted_image_path = '1_un.png'

cam_maker = 'NIKON CORPORATION'
cam_model = 'NIKON D3S'
lens_maker = 'Nikon'
lens_model = 'Nikkor 28mm f/2.8D AF'

db = lensfunpy.Database()
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]

im = cv2.imread(image_path)
height, width = im.shape[0], im.shape[1]

mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(focal_length, aperture, distance)

undist_coords = mod.apply_geometry_distortion()
im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
cv2.imwrite(undistorted_image_path, im_undistorted)