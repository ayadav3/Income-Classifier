__author__ = 'amityadav'


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
    Normalizes the features data set according to the option provided in the num variable.

    :param X: a data frame with all the feature variables and data
    :param num: takes value 1 or 2. If num equals 1, then use scale(). If num equals 2, then use MinMaxScaler().
    :return: a normalized data frame
    """
    if num == 1:
        normX = scale(X)
    elif num == 2:
        min_max_scaler = MinMaxScaler()
        normX = min_max_scaler.fit_transform(X)
    else:
        print 'You have selected wrong value for "num" parameter. It can take either 1 or 2 as a valid value.'

    return normX


def decisionTree(X, y, X_test, y_test):
    """
    Implements the decision tree classifier from the sklearn package.

    :param X: a training data set containing the feature values
    :param y: an array of training target variables
    :param X_test: a test data set containing the feature values
    :param y_test: an array of testing set target variables
    :return: prints the f1_score
    """

    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    print f1_score(y_test, ypred, average=None)


def randomForest(X, y, X_test, y_test):
    """
    Implements the random forest classifier from the sklearn package.

    :param X: a training data set containing the feature values
    :param y: an array of training target variables
    :param X_test: a test data set containing the feature values
    :param y_test: an array of testing set target variables
    :return: prints the f1_score
    """

    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=33, n_jobs=-1, max_features=50)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    print f1_score(y_test, ypred, average=None)
    # print cross_val_score(clf, X, y, cv = 3, scoring = 'accuracy').mean()


def logistic(X, y, X_test, y_test):
    """
    Implements the logistic regression classifier from the sklearn package.

    :param X: a training data set containing the feature values
    :param y: an array of training target variables
    :param X_test: a test data set containing the feature values
    :param y_test: an array of testing set target variables
    :return: prints the f1_score
    """

    clf = LogisticRegression(penalty='l2', C=0.5, random_state=33)
    clf.fit(X, y)
    ypred = clf.predict(X_test[:])
    # print clf.score(X, y)
    print f1_score(y_test, ypred, average=None)
    #print cross_val_score(clf, X, y, cv = 3, scoring = 'accuracy').mean()


def KNN(X, y, X_test, y_test):
    """
    Implements the K-Nearest Neighbor classifier from the sklearn package.

    :param X: a training data set containing the feature values
    :param y: an array of training target variables
    :param X_test: a test data set containing the feature values
    :param y_test: an array of testing set target variables
    :return: prints the cross_val_score
    """

    clf = KNeighborsClassifier(n_neighbors=5)
    print cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()


def SVM(X, y, X_test, y_test):
    """
    Implements the Support Vector Machine classifier from the sklearn package.

    :param X: a training data set containing the feature values
    :param y: an array of training target variables
    :param X_test: a test data set containing the feature values
    :param y_test: an array of testing set target variables
    :return: prints the f1_score
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

    # Reading the original training data as a pandas data frame
    data = pd.read_csv('census-income.data', header=None, delimiter=',')

    # Reading the test data as a pandas data frame
    data_test = pd.read_csv('census-income.test', header=None, delimiter=',')

    # Naming columns of the training and test data set as per the code book
    data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                    'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                    'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                    'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                    'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                    'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                    'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR',
                    'target']
    data_test.columns = data.columns

    # Making a list of columns having continuous values. It includes the target variable also.
    continuous_columns = ['AAGE', 'AHRSPAY', 'DIVVAL', 'NOEMP', 'CAPGAIN', 'CAPLOSS', 'WKSWORK', 'MARSUPWT',
                          'target']

    # Making a list of columns having binary values
    binary_columns = ['ASEX']

    data = drop_columns(data)
    data = drop_rows(data)
    data_test = drop_columns(data_test)
    data_test = drop_rows(data_test)

    # List of names of columns with dummy features
    dummy_columns = [col for col in data.columns.values if col not in continuous_columns if
                     col not in binary_columns]

    # Transform the training and test dataset
    data = data_transformation(data, continuous_columns, dummy_columns, binary_columns)
    data_test = data_transformation(data_test, continuous_columns, dummy_columns, binary_columns)

    # Split features and target variables
    X, y = original_data(data)
    X_test, y_test = original_data(data_test)

    # Separate the indices of target between values of 1 and 0
    pos_index, neg_index = np.where(y == 1)[0], np.where(y == 0)[0]

    # Correct the imbalance of the training data set
    indices = sampling(pos_index, neg_index, 50000)

    # Get the final data set for model fitting, this is for training set only, shuffling the data
    X, y = X[indices], y[indices]

    # Applying various algorithms
    logistic(dataNormalization(X, 2), y, dataNormalization(X_test, 2), y_test)
    KNN(X, y, X_test, y_test)
    randomForest(X, y, X_test, X, y, X_test, y_test)
    decisionTree(X, y, X_test, y_test)
    SVM(dataNormalization(X, 2), y)