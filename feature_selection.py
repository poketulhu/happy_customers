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



def remove_features_manually(train, test):
  print(train.shape)
  print(test.shape)

  nulls_train = (train.isnull().sum()==1).sum()
  print('{} nulls in train'.format(nulls_train))



  print(train.shape)



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