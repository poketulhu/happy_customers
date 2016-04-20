import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.preprocessing import Binarizer, scale


def get_most_important_features(train):
  train = train.drop('ID', 1)
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  random_forest = RandomForestClassifier(n_estimators=100)
  random_forest.fit(train_X, train_y)

  feater_importance = pd.Series(random_forest.feature_importances_, index=train_X.columns)
  feater_importance.sort_values(inplace=True)
  feater_importance.tail(20).plot(kind='barh', figsize=(15  ,7), title='Feature importance by random forest')

  # plt.savefig("feature_importance.png")

  grad_boosting = GradientBoostingClassifier()
  grad_boosting.fit(train_X, train_y)

  feater_importance = pd.Series(grad_boosting.feature_importances_, index=train_X.columns)
  feater_importance.sort_values(inplace=True)
  feater_importance.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance by gradient boosting')

  # plt.savefig("feature_importance2.png")

  extra_trees = ExtraTreesClassifier()
  extra_trees.fit(train_X, train_y)

  feater_importance = pd.Series(extra_trees.feature_importances_, index=train_X.columns)
  feater_importance.sort_values(inplace=True)
  feater_importance.tail(20).plot(kind='barh', figsize=(20,7), title='Feature importance by extra trees classifier')

  # plt.savefig("feature_importance3.png")

def num_var4(train):
  print(train.num_var4.describe())

  train.num_var4.hist(bins=100)
  plt.ylabel('Number of customers')
  plt.savefig('num_var4.png')

  sns.FacetGrid(train, hue="TARGET", size=6).map(plt.hist, "num_var4").add_legend()
  plt.savefig('num_var4_2.png')

def var38(train):
#   print(train.var38.describe())
#
#   print(train.loc[train['TARGET']==1, 'var38'].describe())
#
#   train.var38.hist(bins=1000)
#   plt.title('var38 distributed')
#   plt.savefig('var38.png')
#
#   train.var38.map(np.log).hist(bins=1000)
#   plt.title('log var38')
#   plt.savefig('var38_2.png')
#
#   print(train.var38.map(np.log).mode())
#
#   print(train.var38.value_counts())
#   print(train.var38[train['var38'] != 117310.979016494].mean())
#
#   print(train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts())
#   train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100)
#   plt.title('distribution without most common value')
#   plt.savefig('var38_3.png')

  train['var38_mc'] = np.isclose(train.var38, 117310.979016)
  train['var38_log'] = train.loc[~train['var38_mc'], 'var38'].map(np.log)
  train.loc[train['var38_mc'], 'var38_log'] = 0

def var15(train):
  print(train.var15.describe())

  train['var15'].hist(bins=100)
  plt.title('var15 distributed')
  plt.savefig('var15.png')

  sns.FacetGrid(train, hue="TARGET", size=6).map(sns.kdeplot, "var15").add_legend()
  plt.savefig('var15_2.png')

def saldo_var30(train):
  print(train.saldo_var30.describe())

  train.saldo_var30.hist(bins=100)
  plt.savefig('saldo_var30.png')

  train['saldo_var30_log'] = train.saldo_var30.map(np.log)
  sns.FacetGrid(train, hue="TARGET", size=6).map(sns.kdeplot, "saldo_var30_log").add_legend()
  plt.savefig('saldo_var30_2.png')

def select_with_chi2_and_f_classif(train):
  p = 3
  train = train.drop('ID', 1)
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  X_bin = Binarizer().fit_transform(scale(train_X))
  selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, train_y)
  selectF_classif = SelectPercentile(f_classif, percentile=p).fit(train_X, train_y)

  chi2_selected = selectChi2.get_support()
  chi2_selected_features = [ f for i,f in enumerate(train_X.columns) if chi2_selected[i]]
  print('Chi2 отобрал {} признаков {}.'.format(chi2_selected.sum(),
     chi2_selected_features))
  f_classif_selected = selectF_classif.get_support()
  f_classif_selected_features = [ f for i,f in enumerate(train_X.columns) if f_classif_selected[i]]
  print('F_classif отобрал {} признаков {}.'.format(f_classif_selected.sum(),
     f_classif_selected_features))
  selected = chi2_selected & f_classif_selected
  print('Chi2 & F_classif отобрали {} признаков'.format(selected.sum()))
  features = [ f for f,s in zip(train_X.columns, selected) if s]
  return features

def transform_with_chi2_selected(train, test, features):
  new_train = train[['ID'] + features + ['TARGET']]
  new_train.to_csv('train_after_chi2.csv')

  new_test = test[['ID'] + features]
  new_test.to_csv('test_after_chi2.csv')

def select(train, test):
  features = ['saldo_var30', 'var15', 'saldo_var5', 'ind_var30', 'var38']
  new_train = train[['ID'] + features + ['TARGET']]
  new_train.to_csv('train_after_select.csv')

  new_test = test[['ID'] + features]
  new_test.to_csv('test_after_select.csv')

def select_with_rf_and_lr(train, test):
  features = ['saldo_var30', 'var15', 'saldo_var8', 'var38', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult3', 'num_var45_ult3']
  new_train = train[['ID'] + features + ['TARGET']]
  new_train.to_csv('train_after_rf_and_lr.csv')

  new_test = test[['ID'] + features]
  new_test.to_csv('test_after_rf_and_lr.csv')

train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
# get_most_important_features(train)
# num_var4(train)
#var38(train)
# var15(train)
#saldo_var30(train)
# print(select_with_chi2_and_f_classif(train))
# select(train, test)
# transform_with_chi2_selected(train, test, select_with_chi2_and_f_classif(train))
select_with_rf_and_lr(train, test)