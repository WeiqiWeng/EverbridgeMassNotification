import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

main_data = pd.read_pickle('../data/pickle/classification/train_balanced.pickle')

label = list(main_data.attemptState)
main_data.drop('attemptState', axis=1, inplace=True)

print('sample size = %d, feature size = %d' % main_data.shape)

print('tuning svm with rbf kernel ...')
parameters = {'C':np.arange(0.1, 1.1, 0.1), 'gamma':np.arange(0.1, 0.6, 0.1)}
rbf_svm = SVC(kernel='rbf', decision_function_shape='ovr', max_iter=1000)
clf = GridSearchCV(rbf_svm, parameters)
clf.fit(main_data, label)
rbf_svm_score = clf.score(main_data, label)
print('training accuracy for rbf kernel: ', rbf_svm_score)
pickle.dump(clf.best_estimator_, open("../model/rbf_svm.pickle", "wb"))

print('tuning svm with polynomial kernel ...')
parameters = {'C':np.arange(0.1, 1.1, 0.1), 'degree':[1, 2, 3, 4, 5]}
poly_svm = SVC(kernel='poly', decision_function_shape='ovr', max_iter=1000)
clf = GridSearchCV(poly_svm, parameters)
clf.fit(main_data, label)
poly_svm_score = clf.score(main_data, label)
print('training accuracy for poly kernel: ', poly_svm_score)

print('tuning svm with linear kernel ...')
parameters = {'C':np.arange(0.1, 1.1, 0.1)}
linear_svm = LinearSVC(multi_class='ovr', max_iter=1000, dual=False)
clf = GridSearchCV(linear_svm, parameters)
clf.fit(main_data, label)
linear_svm_score = clf.score(main_data, label)
print('training accuracy for linear svm: ', linear_svm_score)
pickle.dump(clf.best_estimator_, open("../model/linear_svm.pickle", "wb"))