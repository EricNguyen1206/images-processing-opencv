import cv2

def edge_detect(img):
  canny = cv2.Canny(img, 100, 200)
  return canny


