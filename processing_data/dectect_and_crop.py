from imutils import paths
import imutils
import cv2
import numpy as np

def sharping(img):
  kernel = np.array([[0, -1,  0],
                    [-1,  5, -1],
                      [0, -1,  0]])
  return cv2.filter2D(img, -1, kernel)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
# imagePath = list(paths.list_images('group'))
imagePath = list(paths.list_images('imgs'))

def detect_multiple(imagePath):
    for i, img in enumerate(imagePath):
        name = img.split('.')[-2]

        img = cv2.imread(img)
        img = imutils.resize(img, height=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for j,(x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = img[y+2:y+h-2, x+2:x+w-2]
            face = cv2.resize(face, (64, 128)) 
            face = sharping(face)
            cv2.imwrite(name + str(i) + str(j) + '.jpg', face)
            # cv2.imshow(name + str(j), face)
        cv2.imshow('img', img)
        cv2.waitKey(500)

detect_multiple(imagePath)

# Convert into grayscale

# Detect faces
# crop = []

# Draw rectangle around the faces
    # face = img[x:x+w, y:y+h]
    # cv2.imshow("face",face)
    # cv2.imwrite(filename=("crop" + str(i) + ".jpg"), img=face)


# Display the output