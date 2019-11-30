# Imported libraries:
# sklearn.preprocessing in order to use LabelEncoder to convert the labels into a binary representation
# sklearn.preprocessing.MinMaxScaler to scale the features so that there is no bias when the model is weighting the features
# sklearn.naive_bayes.GaussianNB to use the Gaussian Naive Bayes model on the data
# sklearn.tree.DecisionTreeClassifier to use the Decision Tree model on the data
# sklearn.model_selection.KFold to prerform ten-fold cross validation on the data and obtain a mean score on each model
# numpy in order to use numpy arrays for efficient data manipulation
# csv to read and write to csv files

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math
import csv
import os

# Instantiation of both the LabelEncoder and the MinMaxScaler:

le = preprocessing.LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 1))


# The function below extracts the data from a .data, .csv or .txt file:


def get_data(filename):
    with open(filename, "rt", encoding='utf8') as f:
        file = csv.reader(f)
        temp = list(file)
    return temp

# The function below writes a list or nested list to a .csv which will be used to export the features and labels
#  for review:


def to_csv(filename, nested_list):
    with open(filename, 'w', newline='\n', encoding='utf-8'):
        output_array = np.array(nested_list)
        np.savetxt(filename, output_array, delimiter=",")


# The function below clean the data by removing unknown data, represented as a ?, with the mean of that feautre
# column:


def clean_with_mean(lists):
    for nested_list in lists:
        for i, item in enumerate(nested_list):
            if item == '?':
                nested_list[i] = np.NaN

    means = np.nanmean(np.array(lists).astype(float), axis=0)

    for nested_list in lists:
        for y, item in enumerate(nested_list):
            if math.isnan(float(item)):
                nested_list[y] = means[y]

    return(np.array(lists))


# The function below seperate the features and labels from the input data and returns both respective
# datasets.


def seperate_features_and_labels(file):
    features = []
    labels = []
    for row in file:
        features.append(row[2:])
        labels.append(row[1])

    labels_encoded = le.fit_transform(labels)
    features = scaler.fit_transform(clean_with_mean(features))

    return(features, labels_encoded)


# The function below performs ten fold cross validation by indexing the features and labels,
# randomly grouping the features and labels into sets of training and test data respectively. Because there are ten
# splits, the data is split into a 90 to 10 ratio of test and train respectively.
# A score is obtained by comparing the prediction of the model with the actual label associated with the features.
# The process is repeated ten times and a mean of the scores is calculated that represents the effectiveness of
# the model in predicting the data.

def ten_fold(model, features, labels):
    scores = []
    splits = 10
    cv = KFold(n_splits=splits, random_state=1, shuffle=False)
    for train_index, test_index in cv.split(features):
        x_train, x_test, y_train, y_test = features[train_index], features[
            test_index], labels[train_index], labels[test_index]
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))
    return scores


while True:
    filename = input(
        'Enter the name of the transaction file. Include the file extension. (eg. \'.txt\') : ')
    if(os.path.exists(filename) == False):
        print('The file you selected does not exist, please try again')
        continue
    else:
        break
print('Process started...')

# Data is extracted from the .data file using the get_data function and seperated using the
# seperate_features_and_labels function. The features and labels are exported to csv files for review:

print('Isolating Features and Labels...')
data = get_data(filename)
features, labels = seperate_features_and_labels(data)
to_csv('Features.csv', features)
to_csv('Labels.csv', labels)

# The Gaussian Naive Bayes model is instantiated and is inputted into the ten_fold function along with the
# features and labels. The scores are returned and the mean of the scores is outputted for review:

print('Applying ten-fold cross validation to the Naive Bayes model on the data...')
nb_model = GaussianNB()
scores = ten_fold(nb_model, features, labels)
nb_mean_score = "{:.1%}".format(np.mean(scores))
print(
    f'The mean score of the Gaussian Naive Bayes Model for the data is {nb_mean_score}')

# The Decision Tree model is instantiated and is inputted into the ten_fold function along with the features
# and labels. The scores are returned and the mean of the scores is outputted for review:

print('Applying ten-fold cross validation to the Decision Tree model on the data...')
dt_model = DecisionTreeClassifier()
scores = ten_fold(dt_model, features, labels)
dt_mean_score = "{:.1%}".format(np.mean(scores))
print(
    f'The mean score of the Decision Tree Model for the data is {dt_mean_score}')

print('Process completed.')
