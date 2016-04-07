import pandas as pd
import sys
import logging
import xgboost as xgb
from sklearn import svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, \
  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


logger = logging.getLogger("Santander Customer Satisfaction")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logging.basicConfig(level = logging.DEBUG)

def predict_for_all_models(classifiers, train):
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, train_size=.60)

  for key, value in classifiers.items():
    print("Predicting for {}...".format(key))
    # print(key)
    model = value.fit(train_X, train_y)
    output = model.predict(test_X)
    print("AUC: {}".format(metrics.roc_auc_score(test_y, output)))
    # print(metrics.roc_auc_score(test_y, output))

def predict_for_one_model(classifier, train, test):
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  ids = test['ID']

  logger.info("Predicting...")
  model = classifier
  model = model.fit(train_X, train_y)
  output = model.predict(test)

  ids = pd.DataFrame(ids, columns=["ID"])
  out = pd.DataFrame(output, columns=['TARGET'])
  result = pd.concat([ids, out], axis=1)
  result.to_csv("simple_prediction.csv")

# base data
# train = pd.read_csv("train.csv")
# train = pd.read_csv("train_after_rlv.csv")
train = pd.read_csv("train_after_manual_selection.csv")
print(train.shape)

# test = pd.read_csv("test.csv")
test = pd.read_csv("test_after_manual_selection.csv")
print(test.shape)

classifiers = {
                "Logistic regression": LogisticRegression(),
                "Naive bayes": GaussianNB(),
                # "K nearest neughbours": KNeighborsClassifier(),
                # "SVM": svm.SVC(),
                "Decision tree": DecisionTreeClassifier(),
                "Extra tree": ExtraTreeClassifier(),
                "Ada boost": AdaBoostClassifier(),
                "Bagging": BaggingClassifier(),
                "Extra trees": ExtraTreesClassifier(),
                "Gradient boosting": GradientBoostingClassifier(),
                "Random forest": RandomForestClassifier()
              }

# predict_for_all_models(classifiers, train)


predict_for_one_model(DecisionTreeClassifier(), train, test)