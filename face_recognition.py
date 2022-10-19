import pickle
import cv2
from data_processing import get_feature
import imutils
from imutils import paths

# Input: là đầu ra của face_detection
# Output: vẽ ảnh với name của các thành viên 
def face_recognition(faces_detect, rect_img): 
  # load model đã train
  model = pickle.load(open("./model.h5", 'rb'))

  # travel các face_detect và predict tên 
  for (x, x_end, y, y_end) in faces_detect:
    crop_img = rect_img[y:y_end, x:x_end]

    fd = get_feature(crop_img, tracking=False)[0]

    name = model.predict([fd])[0]

    cv2.putText(rect_img, name, (x, y_end+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  return rect_img