import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def standardized_data_with_bias():
    breast_cancer = load_breast_cancer()
    #print(breast_cancer.DESCR)
    #Pulling training/testing data from sklearn
    X, t = load_breast_cancer(return_X_y=True)

    # data splitting between train and test
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.2, random_state =10)

    #feature standardization
    # Initialize the StandardScaler
    sc = StandardScaler()

    # Fit and transform the training data
    X_train = sc.fit_transform(X_train)

    # Transform the test data
    X_test = sc.transform(X_test)

    # Insert column of 1s for bias
    X1_train = np.insert(X_train, 0, 1, axis=1)
    X1_test = np.insert(X_test, 0, 1, axis=1)

    return X1_train, t_train, X1_test, t_test


X1_train, t_train, X1_test, t_test = standardized_data_with_bias()


print(X1_train.shape)