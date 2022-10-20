import pickle
import cv2
from data_processing import get_feature
import imutils
from imutils import paths


def predict(file):
  fd, image, _  = get_feature(file)

  name_predict = model.predict([fd])[0]

  image = imutils.resize(image, width=200)
  cv2.putText(image, name_predict, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  cv2.imshow('predicted image', image)
  cv2.waitKey(300)

if __name__ == '__main__':
  model = pickle.load(open("./model.h5", 'rb'))
  image_test_path = list(paths.list_images('test_imgs'))

  for img_name in image_test_path:
    predict(img_name)
