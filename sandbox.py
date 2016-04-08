import numpy as np        

myList = [1,2,3,4,5,6,7,3,4,5,3,5,99]

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
      data[i] = mean
    i += 1

  return data

print(replace_outliers(myList))