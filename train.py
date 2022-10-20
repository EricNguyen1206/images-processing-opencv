import pickle
from sklearn.svm import SVC
from data_processing import extract_feature
from sklearn import metrics


if __name__ == '__main__':
  data, labels = extract_feature('preprocessing_image', tracking=False)
  print('Extract done!')

  # build model
  clf = SVC(kernel='poly', decision_function_shape='ovo', gamma='auto', probability=True)
  
  print('Training...')

  # training
  clf.fit(data, labels)

  print('Train done!')

  # save model
  pickle.dump(clf, open("./model.h5", 'wb'))

  # test
  print('Get data and label test...')
  data_test, labels_test = extract_feature('test_imgs', tracking=False)

  print('Evaluating model...')

  pred = clf.predict(data_test)
  acc = metrics.accuracy_score(labels_test, pred)

  print('Accuracy: '.format(acc))