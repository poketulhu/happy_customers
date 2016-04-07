from itertools import combinations

import numpy as np
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

def identify_constant_features(data):
  count_uniques = data.apply(lambda x: len(x.unique()))
  constants = count_uniques[count_uniques == 1].index.tolist()
  return constants

def remove_constant_features(train):
  print("Число признаков перед удалением признаков-констант: {}".format(train.shape[1] - 1))
  constant_features_train = set(identify_constant_features(train))
  print('{} переменных-констант в тренировочном наборе'.format(len(constant_features_train)))
  train.drop(constant_features_train, inplace=True, axis=1)
  print("Число признаков после удаления признаков-констант: {}".format(train.shape[1] - 1))

def identify_equal_features(data):
  features_to_compare = list(combinations(data.columns.tolist(),2))
  equal_features = []
  for compare in features_to_compare:
    is_equal = np.array_equal(data[compare[0]],data[compare[1]])
    if is_equal:
      equal_features.append(list(compare))
  return equal_features

def remove_equal_features(train):
  print("Число признаков перед удалением одинаковых признаков: {}".format(train.shape[1] - 1))
  equal_features_train = identify_equal_features(train)
  print('{} пар одинаковых признаков в тренировочном наборе'.format(len(equal_features_train)))
  features_to_drop = np.array(equal_features_train)[:,1]
  train.drop(features_to_drop, axis=1, inplace=True)
  print("Число признаков после удаления одинаковых признаков: {}".format(train.shape[1] - 1))

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# data_description(train, test)
remove_constant_features(train)
remove_equal_features(train)
# train = remove_nans(train)
# train.to_csv("train_after_processing.csv")
