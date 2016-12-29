import data_preprocessing_model as dpm
import constants
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split


df = pd.read_pickle('../data/pickle/everbridge_data_raw.pickle')

df_1 = df[(df.attemptState == 1) | (df.attemptState == 2)]
df_1['duration'] = np.zeros(df_1.shape[0])

df.callStartDate = df.callStartDate.astype(str)
df_1.callStartDate = df_1.callStartDate.astype(str)

n = 0

for i, row_1 in df_1.iterrows():
    tmp = df[(df.id == row_1['id']) & (df.pathPrompt == row_1['pathPrompt'])]
    time1 = datetime.datetime.strptime(row_1.callStartDate, '%Y-%m-%d %H:%M:%S')
    duration = 0
    n += 1
    for j, row_2 in tmp.iterrows():
        time2 = datetime.datetime.strptime(row_2.callStartDate, '%Y-%m-%d %H:%M:%S')
        duration = max((time2 - time1).seconds, duration)
    if duration > 0:
        df_1.loc[(df_1.id == row_1['id']) &
                 (df_1.pathPrompt == row_1['pathPrompt']) &
                 (df_1.callStartDate == row_1['callStartDate']), 'duration'] = duration
    print('matching sample ', n)

df_1 = df_1[0 < df_1.duration]

df_1.drop(['callStartDate', 'attempt', 'id'], axis=1, inplace=True)
df_1 = dpm.bash_add_dummy_indicator(df_1, constants.nominal_features)
df_1 = dpm.fix_skewness(df_1, constants.numerical_features[1:])

df_1.drop('tmp_ones', axis=1, inplace=True)

train, test = train_test_split(df_1, test_size = 0.4)
validation, test = train_test_split(test, test_size = 0.5)

print('training data set size: ',len(train))
print('validation data set size: ',len(validation))
print('test data set size: ',len(test))

train.to_csv('../data/csv/train_prediction.csv')
# train.to_pickle('../data/pickle/prediction/train_prediction.pickle')

validation.to_csv('../data/csv/validation_prediction.csv')
# validation.to_pickle('../data/pickle/prediction/validation_prediction.pickle')

test.to_csv('../data/csv/test_prediction.csv')
# test.to_pickle('../data/pickle/prediction/test_prediction.pickle')



