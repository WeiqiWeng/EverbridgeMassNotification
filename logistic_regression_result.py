import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import cross_val_score

validation_data = pd.read_pickle('../data/pickle/classification/validation_balanced.pickle')
validation_label = list(validation_data.attemptState)
validation_data.drop('attemptState', axis=1, inplace=True)

clf = pickle.load(open( "../model/logistic.pickle", "rb" ))

folds = range(5, 21)
scores = []
std = []
for n in folds:
    print('%d folds for model' % n)
    result = cross_val_score(clf, validation_data, validation_label, cv=n)
    scores.append(np.mean(result))
    std.append(np.std(result))

plt.errorbar(folds, scores, yerr=std, label='Softmax')
plt.xlim((0, 25))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pics/error_bar_softmax.png')
plt.close()