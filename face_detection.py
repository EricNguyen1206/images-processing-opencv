import cv2

# Input: ảnh đã đc cân bằng sáng, lọc nhiễu
# Output: (crop_img, x, x_end, y, y_end)
def face_detect(img):
    # load cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # chuyển thành ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # nhận diện
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    out_put = []

    # vẽ hình vuông nhận diện và trả kết quả
    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      out_put.append((x, x + w, y, y + h))

    cv2.imshow('face_detection', img)

    return out_put, img