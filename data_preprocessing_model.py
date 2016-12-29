import constants
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.style.use('ggplot')


def extract_notification_id(element):

    t = element.partition('_')
    return t[0]


def extract_contact_id(element):

    t = element.partition('_')
    return t[-1]


def add_dummy_indicator(data_frame, indicator):

    value_list = pd.unique(data_frame[indicator])
    print('%s "\n" has %d different values' % (value_list, len(value_list)))
    for item in value_list[0:-1]:
        dummy_ind_name = str(indicator) + "_" + str(item)
        data_frame[dummy_ind_name] = data_frame['tmp_ones'].where(data_frame[indicator] == item, other=0)

    data_frame.drop([indicator], axis=1, inplace=True)
    return data_frame


def transform_feature_datatype(df, features, type):

    for i in features:
        df[i] = df[i].astype(type)

    return df


def bash_add_dummy_indicator(data_frame, nominal_features):

    for feature in nominal_features:
        data_frame = add_dummy_indicator(data_frame, feature)

    return data_frame


def fix_skewness(df, features):

    sample_size = df.shape[0]
    for feature in features:
        df[feature] = np.log(df[feature])
        df[feature] += np.random.normal(0, 1, sample_size)

    df[features] = preprocessing.scale(df[features])

    return df


def hist_numerical_features(main_data, numerical_features):

    for feature in numerical_features:

        n, bins, patches = plt.hist(main_data[feature], normed=1, bins=20, alpha=0.75)
        y = mlab.normpdf(bins, 0, 1)
        l = plt.plot(bins, y, 'y--', linewidth=1)
        plt.xlabel('$log_e$('+ feature + ')')
        plt.ylabel('frequency')
        plt.title(r'histogram of $log_e$(' + feature + ')')
        plt.grid(True)
        plt.savefig('../pics/' + feature + '_hist.png')
        plt.close()