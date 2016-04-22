import pandas as pd
import sys
import logging
from sklearn import svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, \
  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


logger = logging.getLogger("Santander Customer Satisfaction")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logging.basicConfig(level = logging.DEBUG)

def simple_prediction():
  train = pd.read_csv('train.csv')
  test = pd.read_csv('test.csv')

  ids = test['ID']
  train = train.drop('ID', 1)
  test = test.drop('ID', 1)
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  model = GradientBoostingClassifier(n_estimators=20, max_features=1, max_depth=3,
                                                        min_samples_leaf=100, learning_rate=0.1,
                                                        subsample=0.65, loss='deviance', random_state=1)
  model.fit(train_X, train_y)
  prediction = model.predict_proba(test)[:, 1]
  prediction = pd.DataFrame(prediction)
  output = pd.concat([ids, prediction], axis=1)
  output.to_csv("simple.csv")

def predict_for_all_models(classifiers, train):
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, train_size=.60)

  for key, value in classifiers.items():
    print("Predicting for {}...".format(key))
    model = value.fit(train_X, train_y)
    output = model.predict_proba(test_X)[:, 1]
    print("AUC: {}".format(metrics.roc_auc_score(test_y, output)))

def predict_for_one_model(classifier, train, test):
  train_y = train['TARGET']
  train_X = train.drop('TARGET', 1)

  ids = test['ID']

  logger.info("Predicting...")
  model = classifier
  model = model.fit(train_X, train_y)
  output = model.predict_proba(test)[:, 1]

  ids = pd.DataFrame(ids, columns=["ID"])
  out = pd.DataFrame(output, columns=['TARGET'])
  result = pd.concat([ids, out], axis=1)
  result.to_csv("simple_prediction.csv")

# base data
train = pd.read_csv("train.csv")

#chi2 && f_classif selected
train = pd.read_csv('train_after_chi2.csv')
test = pd.read_csv('test_after_chi2.csv')

#selected
train = pd.read_csv('train_after_select.csv')
test = pd.read_csv('test_after_select.csv')

#rf and lr
train = pd.read_csv('train_after_rf_and_lr.csv')
test = pd.read_csv('test_after_rf_and_lr.csv')

#fpr
train = pd.read_csv('train_after_fpr.csv')
test = pd.read_csv('test_after_fpr.csv')

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

predict_for_one_model(GradientBoostingClassifier(), train, test)

# simple_prediction()