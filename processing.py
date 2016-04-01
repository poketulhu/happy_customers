import pandas as pd
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

train = pd.read_csv("train.csv")
train = remove_nans(train)
train.to_csv("train_after_processing.csv")
