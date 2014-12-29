__author__ = 'amityadav'


# Import libraries and packages

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, scale
from sklearn.metrics import confusion_matrix, f1_score


# # get the data from the whole data set and make the predictor as -1 or 1


def drop_columns(data):
    """
    Drops the unimportant columns from the original data

    :param data: a pandas data frame of the original data
    :return: a pandas data frame with only the important columns
    """
    output_data = data.drop(['MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSUN', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'YEAR'], axis=1)

    return output_data


def drop_rows(data):
    """
    Drops rows having missing or NA values from the data set
    The training set is huge, so dropping few rows will not affect the learning
    ability of the classifiers

    :param data: a pandas data frame of the original data
    :return: a pandas data frame without any missing or NA value
    """
    row1 = np.where(data['GRINST'] == ' ?')[0]
    row2 = np.where(data['HHDFMX'] == ' Grandchild <18 ever marr not in subfamily')[0]
    rownum = np.concatenate((row1, row2))
    data = data.drop(data.index[rownum])

    return data


def data_transformation(data, continous, dummy, binary):
    """
    Encodes categorical variables into numerical values such as 0 and 1 using OneHotEncoder function.

    Sklearn functions do not directly accept categorical data in string fromat. Thus, encoding of categorical
    data is required.
    Also, if a feature has only binary values(such as: Gender), then 0,1 encoding is fine. However, if the
    feature has multiple levels, then simple encoding as 0,1,2,3... will not work because it will bring
    additional information about magnitude between different levels.

    :param data: a pandas data frame containing all the variables.
    :param continous: a list of column names(variables) which have continuous values
    :param dummy: a list of column names(variables) which have dummy values
    :param binary: a list of column names(variables) which have binary values
    :return: a transformed data set
    """
    le = LabelEncoder()
    for col1 in dummy:
        le.fit(data[col1])
        data[col1] = le.transform(data[col1])
    dummydata = np.array(data[dummy])
    enc = OneHotEncoder()
    enc.fit(dummydata)
    dummydata = enc.transform(dummydata).toarray()

    for col2 in binary:
        le.fit(data[col2])
        data[col2] = le.transform(data[col2])
    binarydata = np.array(data[binary])

    le.fit(data['target'])
    data['target'] = le.transform(data['target'])
    continuousdata = np.array(data[continous])

    return np.concatenate((dummydata, binarydata, continuousdata), axis=1)