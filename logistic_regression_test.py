import pickle
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

test_data = pd.read_pickle('../data/pickle/classification/test_balanced.pickle')
test_label = list(test_data.attemptState)
test_data.drop('attemptState', axis=1, inplace=True)

validation_data = pd.read_pickle('../data/pickle/classification/validation_balanced.pickle')
train_data = pd.read_pickle('../data/pickle/classification/train_balanced.pickle')
main_data = pd.concat([train_data, validation_data])
label = list(main_data.attemptState)
main_data.drop('attemptState', axis=1, inplace=True)

logreg = lm.LogisticRegression(C = 0.4, penalty='l2', solver='sag', max_iter=1000, multi_class='multinomial')
logreg.fit(main_data, label)
score = logreg.score(test_data, test_label)
print('testing accuracy: ', score)
print('coefficient: ', logreg.coef_)
print('intercept: ', logreg.intercept_)

