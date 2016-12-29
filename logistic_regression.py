import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV

main_data = pd.read_pickle('../data/pickle/classification/train_balanced.pickle')

label = list(main_data.attemptState)
main_data.drop('attemptState', axis=1, inplace=True)

print('sample size = %d, feature size = %d' % main_data.shape)

parameters = {'C':np.arange(0.1, 1.1, 0.1)}
logreg = lm.LogisticRegression(penalty='l2', solver='sag', max_iter=1000, multi_class='multinomial')
clf = GridSearchCV(logreg, parameters)
clf.fit(main_data, label)
print('coefficient: ', clf.best_estimator_.coef_)
print('intercept: ', clf.best_estimator_.intercept_)

print('best hyper-parameter: ', clf.best_params_)
print('training accuracy: ', clf.score(main_data, label))

pickle.dump(clf.best_estimator_, open("../model/logistic.pickle", "wb"))