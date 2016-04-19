import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt

def get_most_important_features(train):
  train = train.drop('ID', 1)
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  # random_forest = RandomForestClassifier(n_estimators=100)
  # random_forest.fit(train_X, train_y)
  #
  # feater_importance = pd.Series(random_forest.feature_importances_, index=train_X.columns)
  # feater_importance.sort_values(inplace=True)
  # feater_importance.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance by random forest')
  #
  # plt.savefig("feature_importance.png")
  #
  # grad_boosting = GradientBoostingClassifier()
  # grad_boosting.fit(train_X, train_y)
  #
  # feater_importance = pd.Series(grad_boosting.feature_importances_, index=train_X.columns)
  # feater_importance.sort_values(inplace=True)
  # feater_importance.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance by gradient boosting')
  #
  # plt.savefig("feature_importance2.png")

  extra_trees = ExtraTreesClassifier()
  extra_trees.fit(train_X, train_y)

  feater_importance = pd.Series(extra_trees.feature_importances_, index=train_X.columns)
  feater_importance.sort_values(inplace=True)
  feater_importance.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance by extra trees classifier')

  plt.savefig("feature_importance3.png")

train = pd.read_csv("train_after_processing.csv")
get_most_important_features(train)