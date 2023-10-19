import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
#print(breast_cancer.DESCR)
#Pulling training/testing data from sklearn
X, t = load_breast_cancer(return_X_y=True)

# data splitting between train and test
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.2, random_state =10)
new_col=np.ones(len(X_train))
X1_train = np.insert(X_train, 0, new_col, axis=1)
new_col=np.ones(len(X_test))
X1_test = np.insert(X_test, 0, new_col, axis=1)

print(X1_train.shape)
print(X1_test.shape)
