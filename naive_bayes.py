import csv
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

le = preprocessing.LabelEncoder()
model = GaussianNB()


def get_labels(filename):
    with open(filename, "rt", encoding='utf8') as f:
        file = csv.reader(f)
        temp = list(file)
    return temp

def seperate_features_and_labels(file):
    features = []
    labels = []

    for row in file:
        row = [float(i.replace('?','0')) for i in row]
        features.append(tuple(row[1:]))
        labels.append(row[-1])

    labels_encoded = le.fit_transform(labels)
    return(features, labels_encoded)


data = get_labels('breast-cancer-wisconsin.data')
features, labels = seperate_features_and_labels(data)
model.fit(features, labels)

predicted = model.predict([[2, 1, 1, 1, 2, 1, 1, 1, 10, 3]])

print("Predicted Value:", predicted)
