
import cv2
import numpy as np
from skimage import feature
import pickle
import imutils
from imutils import paths
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC

import matplotlib.pyplot as plt

def get_feature(image):

  image = cv2.resize(image, (64, 128)) #resize

  #check
  # cv2.imshow('img', image)
  # cv2.waitKey()

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # get feature, origin image and gray img
  result = feature.hog(
      gray, orientations=9, pixels_per_cell=(8, 8),
      cells_per_block=(2, 2), block_norm="L2",
      visualize=True)

  return result[0], image, result[1]

def extract_feature():
  imagePath = list(paths.list_images('preprocessing_image'))
  data = []
  labels = []

  print('Processing...')

  for (_, ip) in enumerate(imagePath):
    name = ip.split(os.path.sep)[-2]
    print(name)
    
    fd, _, gray = get_feature(ip)

    data.append(fd)
    labels.append(name)
  print('extract_done')

  return data, labels

if __name__ == '__main__':
  data, labels = extract_feature()

  # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state = np.random.RandomState())

  # clf = SVC()
  # clf = SVC(kernel='poly', degree=3, C=1)
  clf = SVC(kernel='poly', decision_function_shape='ovr', gamma='auto')
  

  print('train...')

  # clf.fit(list(map(lambda d: d['fd'], data)), labels)

  clf.fit(data, labels)

  # clf = SVC(kernel='rbf', gamma=0.5, C=0.1)

  # clf.fit(data, labels)

  print('train done!')

  pickle.dump(clf, open("./model.h5", 'wb'))

  # predicted = clf.predict([get_feature(os.path.join(os.getcwd(), 'test_imgs/crop-group61.jpg'))[0]])

  # print(predicted)


# load model here
# loaded_model = pickle.load(open("./model.h5", 'rb'))

# clf.save('test.h5')

# for i, p in enumerate(predicted):
#   plt.figure()
#   plt.imshow(x_test[i]['img'])
#   plt.title(p)

# plt.show()

  
