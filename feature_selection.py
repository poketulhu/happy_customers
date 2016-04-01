from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def remove_low_variance(train, test):
  print(train.shape)
  y = train['TARGET']
  sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  train = sel.fit_transform(train)
  print(train.shape)
  train = pd.DataFrame(train)
  train['TARGET'] = y
  train.to_csv("train_after_rlv.csv", index=False)

def identify_constant_features(data):
    count_uniques = data.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

def identify_equal_features(data):
    features_to_compare = list(combinations(data.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = np.array_equal(data[compare[0]],data[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features

def remove_features_manually(train, test):
  print(train.shape)
  print(test.shape)

  nulls_train = (train.isnull().sum()==1).sum()
  print('{} nulls in train'.format(nulls_train))

  constant_features_train = set(identify_constant_features(train))

  print('{} constant features in train'.format(len(constant_features_train)))

  train.drop(constant_features_train, inplace=True, axis=1)

  print(train.shape)

  equal_features_train = identify_equal_features(train)

  print('{} pairs of equal features in train'.format(len(equal_features_train)))

  features_to_drop = np.array(equal_features_train)[:,1]
  train.drop(features_to_drop, axis=1, inplace=True)

  print(train.shape)

  train.to_csv("train_after_manual_selection.csv", index=False)

  y_name = 'TARGET'
  feature_names = train.columns.tolist()
  feature_names.remove(y_name)

  test_names = test.columns.tolist()
  test_names_to_drop = list(set(test_names) - set(feature_names))
  test = test.drop(test_names_to_drop, axis=1)

  test.to_csv("test_after_manual_selection.csv", index=False)


# train = pd.read_csv("train.csv")
train = pd.read_csv("train_after_manual_selection.csv")
test = pd.read_csv("test.csv")
# remove_low_variance(train, test)
# remove_features_manually(train, test)