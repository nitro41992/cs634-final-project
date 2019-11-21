import csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

le = preprocessing.LabelEncoder()
model = DecisionTreeClassifier()
scaler = MinMaxScaler(feature_range=(0, 1))


def get_data(filename):
    with open(filename, "rt", encoding='utf8') as f:
        file = csv.reader(f)
        temp = list(file)
    return temp


def to_csv(filename, nested_list):
    with open(filename, 'w', newline='\n', encoding='utf-8'):
        output_array = np.array(nested_list)
        np.savetxt(filename, output_array, delimiter=",")


def seperate_features_and_labels(file):
    features = []
    labels = []
    for row in file:
        features.append(tuple(le.fit_transform(row[2:])))
        labels.append(row[1])

    labels_encoded = le.fit_transform(labels)
    features = scaler.fit_transform(features)

    return(features, labels_encoded)


data = get_data('wpbc.data')
feature_cols = 'radius (mean of distances from center to points on the perimeter)', 'texture (standard deviation of gray-scale values)', 'perimeter,area,smoothness (local variation in radius lengths)', 'compactness (perimeter^2 / area - 1.0)', 'concavity (severity of concave portions of the contour)', 'concave points (number of concave portions of the contour)', 'symmetry,fractal dimension ("coastline approximation" - 1)'
features, labels = seperate_features_and_labels(data)
to_csv('Features.csv', features)
to_csv('Labels.csv', labels)

scores = []
splits = 10
cv = KFold(n_splits=splits, random_state=1, shuffle=False)
for train_index, test_index in cv.split(features):

    x_train, x_test, y_train, y_test = features[train_index], features[
        test_index], labels[train_index], labels[test_index]
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))

print(np.mean(scores))
