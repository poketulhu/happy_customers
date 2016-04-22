import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from itertools import combinations

def binary_features(data):
  col_names = []
  for col in list(set(data.columns) - set(['TARGET'])):
    le = preprocessing.LabelEncoder()
    le = le.fit(data[col])
    classes = le.classes_
    if len(classes) == 2 and classes[0] == 0 and classes[1] == 1:
      col_names.append(col)
  return col_names

def quantitative_features(data):
  col_names = []
  for col in list(set(data.columns) - set(['TARGET'])):
    le = preprocessing.LabelEncoder()
    le = le.fit(data[col])
    classes = le.classes_
    if classes.dtype == 'float64':
      col_names.append(col)
  return col_names

def data_description(train, test):
  print("Тренировочный набор содержит {} записей.".format(train.shape[0]))
  # Тренировочный набор содержит 76020 записей.
  print("Тестовый набор содержит {} записей.".format(test.shape[0]))
  # Тестовый набор содержит 75818 записей.
  print("Данные содержат {} признаков.".format(train.shape[1] - 1))
  # Данные содержат 370 признаков.
  print("Из них бинарных: {}".format(len(binary_features(train))))
  # Из них бинарных: 66
  print("Из них количественных: {}".format(len(quantitative_features(train))))
  # Из них количественных: 111
  print("Из них номинальных: {}".format(train.shape[1] - 1 - len(binary_features(train)) - len(quantitative_features(train))))
  # Из них номинальных: 193
  df = pd.DataFrame(train.TARGET.value_counts())
  df['Percentage'] = 100*df['TARGET']/train.shape[0]
  print(df)

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
  return train

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
  return train

def replace_outliers(data):
  mean_duration = np.mean(data)
  std_dev_one_test = np.std(data)

  without_outliers = []
  for d in data:
    if abs(d - mean_duration) <= std_dev_one_test:
      without_outliers.append(d)

  mean = np.mean(without_outliers)

  i = 0
  for d in data:
    if abs(d - mean_duration) > std_dev_one_test:
      data.values[i] = mean
    i += 1

  return data

def features_with_outliers(data):
  columns = []
  for col in list(set(data.columns) - set(['ID', 'TARGET'])):
    before = np.mean(data[col])
    data[col] = replace_outliers(data[col])
    after = np.mean(data[col])
    if before != after:
      columns.append(col)
  print("{} признаков с выбросами".format(len(columns)))
  return data

def normalize_data(data, columns):
  for col in columns:
    data[col].values = preprocessing.normalize(data[col].values)
  return data

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# data_description(train, test)
train = remove_constant_features(train)
train = remove_equal_features(train)
# features_with_outliers(train)
train.to_csv('train_after_processing.csv')


test = remove_constant_features(test)
test = remove_equal_features(test)
test.to_csv('test_after_processing.csv')
# train = normalize_data(train, quantitative_features(train))
# train.to_csv("train_after_remove_outliers.csv")
# test = features_with_outliers(test)
# test.to_csv("test_after_remove_outliers.csv")
