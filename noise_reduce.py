import cv2
import cv2

# de_noise_img = cv2.GaussianBlur(noisy_img, (3, 3), 6, 6) # for poisson and gaussian
# de_noise_img = cv2.medianBlur(de_noise_img, 3) # for s&p
# de_noise_img = cv2.fastNlMeansDenoisingColored(noisy_img,None,10,10,7,21)
# de_noise_img = cv2.bilateralFilter(src=noisy_img, d=5, sigmaColor=100, sigmaSpace=20)

def noise_reduce(type, img):

  if type == None:
    return img

  de_noise_img = None

  if type == 'gauss' or type == 'poisson':
    de_noise_img = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=3, sigmaY=3)
  elif type == 's&p':
    de_noise_img = cv2.medianBlur(src=img, ksize=3)
  elif type == 'bilateral':
    de_noise_img = cv2.bilateralFilter(src=img, d=5, sigmaColor=10, sigmaSpace=30)
  elif type == 'mean':
    de_noise_img = cv2.fastNlMeansDenoisingColored(src=img, h=5, hColor=3, templateWindowSize=7, searchWindowSize=7)

  return de_noise_img

