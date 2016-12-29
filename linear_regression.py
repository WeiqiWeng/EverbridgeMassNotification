import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV


train_data = pd.read_pickle('../data/pickle/train_prediction.pickle')
print(len(train_data))
train_label = list(train_data.duration)
train_label = preprocessing.scale(train_label)
train_data.drop('duration', axis=1, inplace=True)

# train_data_mini = train_data[constants.numerical_features[1:] + ['pathPrompt_Personal Cell Text',
#                                                                            'pathPrompt_Home Phone',
#                                                                            'pathPrompt_Office Email',
#                                                                            'pathPrompt_SMS',
#                                                                            'pathPrompt_Cell Phone',
#                                                                            'pathPrompt_Work Phone',
#                                                                            'priority_Priority',
#                                                                            'type_Standard',
#                                                                            'country_US',
#                                                                            'country_IN',
#                                                                            'country_GB',
#                                                                            'organizationId_892807736722516',
#                                                                            'organizationId_1332612387831898']]
#
reg = lm.LinearRegression()
reg.fit(train_data, train_label)
print(reg.score(train_data, train_label))

# parameters = {'C':np.arange(0.1, 1, 0.1)}
# reg = svm.SVR(kernel='rbf', max_iter=1000)
# clf = GridSearchCV(reg, parameters)
# clf.fit(train_data, train_label)
# print('best parameter: ', clf.best_params_)
# print('accuracy: ', clf.score(train_data, train_label))

