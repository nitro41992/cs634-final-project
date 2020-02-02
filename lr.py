import csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import math


le = preprocessing.LabelEncoder()
le_gender = preprocessing.LabelEncoder()
le_rvsp = preprocessing.LabelEncoder()
le_rv_function = preprocessing.LabelEncoder()
le_size = preprocessing.LabelEncoder()
le_intervention = preprocessing.LabelEncoder()


model = LogisticRegression(solver='lbfgs')
scaler = MinMaxScaler(feature_range=(0, 1))


def get_data(filename):
    with open(filename, "rt", encoding='utf8') as f:
        reader = csv.reader(f)
        next(reader, None)
        file = csv.reader(f)
        temp = list(file)
    return temp


def to_csv(filename, nested_list):
    with open(filename, 'w', newline='\n', encoding='utf-8'):
        output_array = np.array(nested_list)
        np.savetxt(filename, output_array, delimiter=",")


def clean_and_mean(lists):
    lists = lists.replace('N/A', np.NaN)
    num_lists = lists.apply(pd.to_numeric, errors='coerce')
    cleaned_lists = num_lists.apply(lambda x: x.fillna(x.mean()))
    return(np.array(cleaned_lists))


included_cols = [3, 4, 5, 6, 7, 8, 9, 10, 11,
                 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30]


def seperate_features_and_labels(file):
    features = []
    labels = []
    for row in file:
        filt_row = list(row[i] for i in included_cols)
        features.append(filt_row)
        # labels.append(row[16])
        labels.append(row[16])

    labels_encoded = le.fit_transform(labels)

    columns = ['Sex (M/F)', 'BMI DM (1/0)', 'HTN (1/0)', 'COPD (1/0)', 'CTEPH (1/0)', 'ESRD (1/0)', 'Hx of Malignancy (1/0)', 'PE (1/0)', 'Original EDA  (cm2)', 'Original ESA (cm2)',
               ' Original FAC (%)', 'Original EndoGLS (%)', 'Size/Location of Embolus', 'RVSP', 'RV', 'Size', 'RV Function', 'McConnell\'s Sign', 'TR Velocity', 'Intervention']

    df_features = pd.DataFrame(features, columns=columns)
    df_features['Sex (M/F)'] = le_gender.fit_transform(df_features['Sex (M/F)'])
    df_features['RVSP'] = le_rvsp.fit_transform(df_features['RVSP'])
    df_features['Size'] = le_size.fit_transform(df_features['Size'])
    df_features['RV Function'] = le_rv_function.fit_transform(
        df_features['RV Function'])
    df_features['Intervention'] = le_intervention.fit_transform(
        df_features['Intervention'])

    df_features.to_csv(r'df_features.csv')

    features = scaler.fit_transform(clean_and_mean((df_features)))
    return(features, labels_encoded)


data = get_data('scar_data.csv')
features, labels = seperate_features_and_labels(data)
to_csv('Features.csv', features)
to_csv('Labels.csv', labels)


scores = []
splits = 10
cv = KFold(n_splits=splits, shuffle=True)
for train_index, test_index in cv.split(features):

    x_train, x_test, y_train, y_test = features[train_index], features[
        test_index], labels[train_index], labels[test_index]
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))

print(np.mean(scores))
