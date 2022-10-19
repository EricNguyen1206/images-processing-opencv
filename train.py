import pickle
from sklearn.svm import SVC
from data_processing import extract_feature


if __name__ == '__main__':
  data, labels = extract_feature()
  print('Extract done!')

  # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state = np.random.RandomState())

  # build model
  clf = SVC(kernel='poly', decision_function_shape='ovo', gamma='auto')
  
  print('Training...')

  # training
  clf.fit(data, labels)

  print('Train done!')

  # save model
  pickle.dump(clf, open("./model.h5", 'wb'))
