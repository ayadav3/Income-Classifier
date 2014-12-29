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

# Drop the unimportant columns from original data sets
def drop_columns(data):
    data = data.drop(['MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSUN', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'YEAR'], axis=1)

    return data


# Drop rows from original dataset which have missing values
# The training set is huge, so dropping few rows will not affect the learning ability of the classifiers
def drop_rows(data):
    rownum1 = np.where(data['GRINST'] == ' ?')[0]
    rownum2 = np.where(data['HHDFMX'] == ' Grandchild <18 ever marr not in subfamily')[0]
    rownum = np.concatenate((rownum1, rownum2))
    data = data.drop(data.index[rownum])

    return data


# this is the important part to make sklearn load the data
# sklearn do not directly take the categorical data in string format
# If the feature only have binary values( such as: sex), 0,1 encoding will be fine
# if the feature has multiple levels of value, simply coding as 0,1,2,3,...
# will not work, because you will bring additional magnitude informations between
# different levels. Therefore, need to use OneHotEncoder() function
def data_transformation(data, continous, dummy, binary):

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