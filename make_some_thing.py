from data_processing import blur, adjust_gamma
from contrast_stretching import contrast_stretching
import imutils
from imutils import paths
import os
import cv2
from utils.make_noise import make_noise

def make_noise_imgs(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

  img = imutils.resize(img, width=500)

  noisy = make_noise('localvar', img)

  cv2.imshow('img', noisy)

  cv2.waitKey()

  cv2.imwrite('localvar_noise.jpg', noisy)


# img_path = list(paths.list_images('preprocessing_image/Tri'))

# for i, ip in enumerate(img_path):
#   name =  ip.split(os.path.sep)[-2]
#   img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
#   new_img = blur(img)

#   # cv2.imshow('img', new_img)
#   # cv2.waitKey(500)

#   cv2.imwrite('{0}-blur-{1}.jpg'.format(name, i), new_img)

def read_img_and_resize( file_path, width = 500):
  img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
  return imutils.resize(img, width = width)

if __name__ == '__main__':
  file_path = 'Haar_Cascade/group/group-02.jpg'
  img = read_img_and_resize(file_path)
  new_img = adjust_gamma(img, 2)
  new_img2 = contrast_stretching(new_img)

  cv2.imshow('img', new_img)
  cv2.imshow('img2', new_img2)

  cv2.waitKey()
