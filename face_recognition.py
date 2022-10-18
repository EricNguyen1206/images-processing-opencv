import pickle
import cv2
from data_processing import get_feature
import imutils
from imutils import paths

# load pre-train model

image_test_path = list(paths.list_images('test_imgs'))

# Input: là đầu ra của face_detection
# Output: vẽ ảnh với name của các thành viên 
def predict(file):
  fd, image, gray = get_feature(file)
  name = model.predict([fd])[0]

  image = imutils.resize(image, width=200)
  cv2.putText(image, name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
  cv2.imshow('predicted image', image)
  cv2.waitKey(500)

def face_recognition(faces_detect, rect_img): 
  # load model đã train
  model = pickle.load(open("./model.h5", 'rb'))

  # travel các face_detect và predict nhãn
  for (x, x_end, y, y_end) in faces_detect:
    crop_img = rect_img[y:y_end, x:x_end]

    fd = get_feature(crop_img)[0]

    name = model.predict([fd])[0]

    cv2.putText(rect_img, name, (x, y_end+12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

  cv2.imshow('face_recognition', rect_img)
  cv2.waitKey()