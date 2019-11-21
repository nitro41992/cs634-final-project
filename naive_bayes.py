import csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

le = preprocessing.LabelEncoder()
model = GaussianNB()
scaler = MinMaxScaler(feature_range=(0, 1))

def get_labels(filename):
    with open(filename, "rt", encoding='utf8') as f:
        file = csv.reader(f)
        temp = list(file)
    return temp

def seperate_features_and_labels(file):
    features = []
    labels = []
    for row in file:
       for i in range(len(row)):
            if row[i] == 'M':
                row[i] = 1
            elif row[i] == 'B':
                row[i] = 0
            elif row[i] == 'R':
                row[i] = 1
            elif row[i] == 'N':
                row[i] = 0
            elif row[i] == '?':
                row[i] = 0
       features.append(tuple(row[1:]))
       labels.append(row[-1])

    labels_encoded = le.fit_transform(labels)
    features = scaler.fit_transform(features)

    return(features, labels_encoded)


data = get_labels('wpbc.data')
features, labels = seperate_features_and_labels(data)
# model.fit(features, labels)

scores = []
splits = 10
cv = KFold(n_splits=splits, random_state=42, shuffle=False)
for train_index, test_index in cv.split(features):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index, "\n")

    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[test_index]
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))

print(np.mean(scores))

