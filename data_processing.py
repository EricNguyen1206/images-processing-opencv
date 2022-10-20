import cv2
import numpy as np
from skimage import feature
from imutils import paths
import os

def sharping(img):
  kernel = np.array([[0, -1,  0],
                    [-1,  5, -1],
                      [0, -1,  0]])
  return cv2.filter2D(img, -1, kernel)

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def blur(img):
  kernel = np.ones((3,3), np.float32)/9
  return cv2.filter2D(img, -1, kernel)

def get_feature(img, tracking=True):
  image = cv2.resize(img, (64, 128)) #resize

  #check
  if tracking:
    cv2.imshow('img', image)
    cv2.waitKey(300)

  # convert to gray color
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # get feature, origin image and gray img
  result = feature.hog(
        gray,  # type of image
        orientations=9, # amount of pin for cal feature
        pixels_per_cell=(8, 8), # amount of window slide
        cells_per_block=(2, 2), # amount of cell 
        block_norm="L2", # func of block normalize
        visualize=True # return the feature image of this
    )
  return result[0], image, result[1]

def extract_feature(f_name, tracking=True):
  imagePath = list(paths.list_images(f_name))
  data = []
  labels = []

  print('Processing...')

  len_of_list = len(imagePath)

  for (i, ip) in enumerate(imagePath):
    print('Get feature of image: ' + str(i) + '/' + str(len_of_list))

    name = ip.split(os.path.sep)[-2]

    img = cv2.imread(ip, cv2.IMREAD_UNCHANGED) #Read image

    fd = np.array(get_feature(img, tracking)[0])

    data.append(fd)
    labels.append(name)

  return data, labels

  
