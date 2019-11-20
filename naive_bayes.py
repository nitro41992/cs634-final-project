import csv


def get_labels(filename):
    with open(filename, "rt", encoding='utf8') as f:
        file = csv.reader(f)
        temp = list(file)
    return temp


def seperate_features_and_labels(file):
    features = []
    labels = []
    for row in file:
        features.append(row[1:])
        labels.append(row[-1])

    return(features, labels)


data = get_labels('breast-cancer-wisconsin.data')
features, labels = seperate_features_and_labels(data)
