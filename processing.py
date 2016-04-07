import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


def remove_nans(data):
  columns_with_nans = []
  columns = data.columns.tolist()
  for col in columns:
    if -999999 in data[col].values:
      columns_with_nans.append(col)

  imp = Imputer(missing_values=-999999, strategy='most_frequent', axis=1)
  for col in columns_with_nans:
    data[col] = pd.Series(imp.fit_transform(data[col])[0])
  return data

def binary_features_count(data):
  col_names = []
  for col in list(set(data.columns) - set(['TARGET'])):
    le = preprocessing.LabelEncoder()
    le = le.fit(data[col])
    classes = le.classes_
    if len(classes) == 2 and classes[0] == 0 and classes[1] == 1:
      col_names.append(col)
  return len(col_names)

def quantitative_features_count(data):
  col_names = []
  for col in list(set(data.columns) - set(['TARGET'])):
    le = preprocessing.LabelEncoder()
    le = le.fit(data[col])
    classes = le.classes_
    if classes.dtype == 'float64':
      col_names.append(col)
  return len(col_names)

def data_description(train, test):
  print("Тренировочный набор содержит {} записей.".format(train.shape[0]))
  # Тренировочный набор содержит 76020 записей.
  print("Тестовый набор содержит {} записей.".format(test.shape[0]))
  # Тестовый набор содержит 75818 записей.
  print("Данные содержат {} признаков.".format(train.shape[1] - 1))
  # Данные содержат 370 признаков.
  print("Из них бинарных: {}".format(binary_features_count(train)))
  # Из них бинарных: 66
  print("Из них количественных: {}".format(quantitative_features_count(train)))
  # Из них количественных: 111
  print("Из них номинальных: {}".format(train.shape[1] - 1 - binary_features_count(train) - quantitative_features_count(train)))
  # Из них номинальных: 193

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data_description(train, test)
# train = remove_nans(train)
# train.to_csv("train_after_processing.csv")
