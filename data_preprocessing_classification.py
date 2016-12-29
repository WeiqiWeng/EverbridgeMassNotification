import data_preprocessing_model as dpm
import constants
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# merge to get raw data
contact_attempt_df = pd.read_pickle('../data/pickle/contact_attempt_df.pickle')

contact_notification_df = pd.read_pickle('../data/pickle/contact_notification_df.pickle')

contact_path_count_df = pd.read_pickle('../data/pickle/contact_path_count_df.pickle')

notification_result_df = pd.read_pickle('../data/pickle/notification_result_df.pickle')

contact_attempt_df['notificationId'] = contact_attempt_df.id.apply(dpm.extract_notification_id)
contact_attempt_df['contactId'] = contact_attempt_df.id.apply(dpm.extract_contact_id)

contact_attempt_df.drop(['id', 'callResult'], axis=1, inplace=True)
contact_notification_df.drop(['createdDate','firstAttemptTime'], axis=1, inplace=True)
notification_result_df.drop(['createdDate',
                             'confirmedCount',
                             'notConfirmedCount',
                             'unreachableCount',
                             'confirmedLateCount'], axis=1, inplace=True)

contact_path_count_df.rename(columns={'id':'contactId'}, inplace=True)
notification_result_df.rename(columns={'id':'notificationId'}, inplace=True)

contact_attempt_df.contactId = contact_attempt_df.contactId.astype(str)
contact_path_count_df.contactId = contact_path_count_df.contactId.astype(str)
contact_notification_df.contactId = contact_notification_df.contactId.astype(str)

contact_notification_df.notificationId = contact_notification_df.notificationId.astype(str)
notification_result_df.notificationId = notification_result_df.notificationId.astype(str)

notification_result_df.organizationId = notification_result_df.organizationId.astype(str)

main_data_df = pd.merge(contact_attempt_df, contact_path_count_df, on=['contactId'])
# print(result.shape)
main_data_df = pd.merge(main_data_df, contact_notification_df, on=['contactId', 'notificationId'])
main_data_df.organizationId = main_data_df.organizationId.astype(str)
# print(result.shape)
main_data_df = pd.merge(main_data_df, notification_result_df, on=['organizationId', 'notificationId'])

main_data_df.drop(['notificationId', 'contactId'], axis=1, inplace=True)

main_data_df = dpm.transform_feature_datatype(main_data_df, constants.nominal_features, str)
main_data_df = dpm.transform_feature_datatype(main_data_df, ['attemptState', 'id', 'callStartDate'], str)

main_data_df.callStartDate = pd.to_datetime(main_data_df.callStartDate, infer_datetime_format=True)
main_data_df = dpm.transform_feature_datatype(main_data_df, constants.numerical_features, float)

main_data_df['tmp_ones'] = pd.Series(np.ones(main_data_df.shape[0]))

# main_data_df[constants.numerical_features] = preprocessing.scale(main_data_df[constants.numerical_features])

main_data_df = main_data_df[main_data_df.attemptState != 'Unreachable']

main_data_df.loc[main_data_df.attemptState == 'Confirmed', 'attemptState'] = 1
main_data_df.loc[main_data_df.attemptState == 'ConfirmedLate', 'attemptState'] = 2
main_data_df.loc[(main_data_df.attemptState == 'Attempted') |
                 (main_data_df.attemptState == 'Unreachable'), 'attemptState'] = 0

main_data_df.to_csv('../data/csv/everbridge_data_raw.csv')
main_data_df.to_pickle('../data/pickle/everbridge_data_raw.pickle')

main_data_df.sort_values(by=['id','attempt'], ascending=False, inplace=True)

main_data_df.drop_duplicates(['id', 'pathPrompt'], keep='first', inplace=True)

main_data_df.drop_duplicates(['id', 'pathPrompt', 'callStartDate'], keep='first', inplace=True)

main_data_df = dpm.bash_add_dummy_indicator(main_data_df, constants.nominal_features)

main_data_df.drop(['tmp_ones', 'id', 'attempt', 'callStartDate'], axis=1, inplace=True)

main_data_df = dpm.fix_skewness(main_data_df, constants.numerical_features[1:])

dpm.hist_numerical_features(main_data_df, constants.numerical_features[1:])

print('data set size: ', len(main_data_df))
train, test = train_test_split(main_data_df, test_size = 0.4)
print('training data set size: ', len(train))
validation, test = train_test_split(test, test_size = 0.5)
print('validation data set size: ',len(validation))
print('test data set size: ',len(test))

train.to_csv('../data/csv/train_classification_balanced.csv')
train.to_pickle('../data/pickle/classification/train_balanced.pickle')

# test.to_csv('../data/csv/test_classification_balanced.csv')
test.to_pickle('../data/pickle/classification/test_balanced.pickle')

# validation.to_csv('../data/csv/validation_classification_balanced.csv')
validation.to_pickle('../data/pickle/classification/validation_balanced.pickle')
