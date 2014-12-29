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
    output_data = data.drop(['MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSUN', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'YEAR'],
                            axis=1)

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


def data_transformation(data, continuous, dummy, binary):
    """
    Encodes categorical variables into numerical values such as 0 and 1 using OneHotEncoder function.

    Sklearn functions do not directly accept categorical data in string format.
    Thus, encoding of categorical data is required.
    Also, if a feature has only binary values(such as: Gender), then 0,1 encoding is fine. However, if the
    feature has multiple levels, then simple encoding as 0,1,2,3... will not work because it will bring
    additional information about magnitude between different levels.

    :param data: a pandas data frame containing all the variables.
    :param continuous: a list of column names(variables) which have continuous values
    :param dummy: a list of column names(variables) which have dummy values
    :param binary: a list of column names(variables) which have binary values
    :return: a transformed data set
    """
    le = LabelEncoder()

    # Encoding the columns with multiple categorical levels
    for col1 in dummy:
        le.fit(data[col1])
        data[col1] = le.transform(data[col1])

    dummy_data = np.array(data[dummy])
    enc = OneHotEncoder()
    enc.fit(dummy_data)
    dummy_data = enc.transform(dummy_data).toarray()

    # Encoding the columns with binary levels
    for col2 in binary:
        le.fit(data[col2])
        data[col2] = le.transform(data[col2])
    binary_data = np.array(data[binary])

    le.fit(data['target'])
    data['target'] = le.transform(data['target'])
    continuous_data = np.array(data[continuous])

    return np.concatenate((dummy_data, binary_data, continuous_data), axis=1)


def original_data(data):
    """
    Splits feature and target variables

    :param data: a pandas data frame containing feature and target variables
    :return: two data frame containing feature and target variables respectively
    """

    predictor_variables = data[:, :-1]
    target_variable = data[:, -1]

    return predictor_variables, target_variable


def sampling(pos_index, neg_index, size):
    """
    Generates indices to perform sampling of the data. It is required to resolve the issue
    of unbalanced data in the training data set. It is to be noted that random oversampling
    or random under sampling depends on the size of sample.

    :param pos_index: an array of all positive indices
    :param neg_index: an array of all negative indices
    :param size: size of required sample
    :return: an array containing equal number of positive and negative indices
    """
    pos_index, neg_index = np.random.choice(pos_index, size), np.random.choice(neg_index, size)
    indices = np.concatenate((pos_index, neg_index))

    return indices


def dataNormalization(X, num):
    """

    :param X:
    :param num:
    :return:
    """
    if num == 1:
        normX = scale(X)
    elif num == 2:
        min_max_scaler = MinMaxScaler()
        normX = min_max_scaler.fit_transform(X)
    else:
        print 'wrong parameter for data normalization'

    return normX


def decisionTree(X, y, X_test, y_test):
    """

    :param X:
    :param y:
    :param X_test:
    :param y_test:
    :return:
    """
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    print f1_score(y_test, ypred, average=None)


def randomForest(X, y, X_test, y_test):
    """

    :param X:
    :param y:
    :param X_test:
    :param y_test:
    :return:
    """
    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=33, n_jobs=-1, max_features=50)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    print f1_score(y_test, ypred, average=None)
    # print cross_val_score(clf, X, y, cv = 3, scoring = 'accuracy').mean()


def logistic(X, y, X_test, y_test):
    """

    :param X:
    :param y:
    :param X_test:
    :param y_test:
    :return:
    """
    clf = LogisticRegression(penalty='l2', C=0.5, random_state=33)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    # print clf.score(X, y)
    print f1_score(y_test, ypred, average=None)
    #print cross_val_score(clf, X, y, cv = 3, scoring = 'accuracy').mean()


def KNN(X, y, X_test, y_test):
    """

    :param X:
    :param y:
    :param X_test:
    :param y_test:
    :return:
    """
    clf = KNeighborsClassifier(n_neighbors=5)
    print cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()


def SVM(X, y, X_test, y_test):
    """

    :param X:
    :param y:
    :param X_test:
    :param y_test:
    :return:
    """
    clf = SVC(C=10)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    # print clf.score(X, y)
    print f1_score(y_test, ypred, average=None)
    #print clf.score(X_test, y_test)
    #print f1_score(y_test, ypred, average = None)
    #print clf.score(X_test, y_test)
    #print cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy').mean()


if __name__ == '__main__':
    data = pd.read_csv('census-income.data', header=None, delimiter=',')  # read the original data
    data_test = pd.read_csv('census-income.test', header=None, delimiter=',')

    data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                    'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                    'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                    'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                    'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                    'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                    'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR',
                    'target']  # original columns name from the supplementary file of data set

    data_test.columns = data.columns
    # print data_test.shape

    continuous_columns = ['AAGE', 'AHRSPAY', 'DIVVAL', 'NOEMP', 'CAPGAIN', 'CAPLOSS', 'WKSWORK', 'MARSUPWT',
                          'target']  # names of columns with continous value( I put the 'target' also here)
    binary_columns = ['ASEX']  # columns with binary value

    data = drop_columns(data)  # drop some colums
    data = drop_rows(data)  # drop some rows, since some rows have missing value
    data_test = drop_columns(data_test)
    data_test = drop_rows(data_test)

    dummy_columns = [col for col in data.columns.values if col not in continuous_columns if
                     col not in binary_columns]  # names of columns with dummy features

    data = data_transformation(data, continuous_columns, dummy_columns, binary_columns)  # transform the dataset
    data_test = data_transformation(data_test, continuous_columns, dummy_columns, binary_columns)
    X, y = original_data(data)  # split features and target
    X_test, y_test = original_data(data_test)

    pos_index, neg_index = np.where(y == 1)[0], np.where(y == 0)[
        0]  # separate the indices of target between values of 1 and 0
    indices = sampling(pos_index, neg_index, 50000)  # correct the imbalance of the training dataset
    X, y = X[indices], y[
        indices]  # get the final dataset for model fitting, this is for training set only, shuffling the data

    # logistic(dataNormalization(X, 2), y, dataNormalization(X_test, 2), y_test)
    #chooseF(X, y, X_test, y_test)

    #randomForest(X, y, X_test, X, y, X_test, y_test)
    #decisionTree(X, y, X_test, y_test)
    #SVM(dataNormalization(X, 2), y) #dataNormalization(X_test, 2), y_test)
