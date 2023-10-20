import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#🥰🥰🥰🥰🥰🥰🥰🥰

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


def sigmoid(x):
    return 1 / (1+ np.exp(-x))


def classification_calculator(y_true, y_pred):
    # positive when t = 0
    # negative when t = 1

    #true positives : y =p, t=p; their number is denoted TP;
    #false positives : y =p, t=n; their number is denoted FP.
    #true negatives : y =n, t=n; their number is denoted TN
    #false negatives : y =n, t=p; their number is denoted FN.
    #Sorina Dumitrescu (McMaster University) COE 4SL4 Fundamentals of Machine Learning September 20, 2023 8 / 13
    return 0

def batch_gradient_descent(X_train,T_train,alpha,iterations):
    cost_list = []
    N = X_train.shape[1] #number of features
    M = X_train.shape[0] #number of examples
    print(N)
    w = np.zeros(N)
    for i in range(iterations):
        z = np.dot(X_train,w)

        #print(z)

        loss  = (1/M) * np.sum((T_train*np.logaddexp(0,-z) + (1-T_train)*np.logaddexp(0,z)))

        y = sigmoid(z)

        w = w - alpha*(1/M)*np.dot(X_train.T,(y - t_train))
        print(loss)
        cost_list.append(loss)

    return  cost_list, w


X1_train, t_train, X1_test, t_test = standardized_data_with_bias()

iterations = 1000
BGD = batch_gradient_descent(X1_train,t_train,0.1,iterations)


Z = np.dot(X1_test,BGD[1])

np.sort(Z)


for beta in Z:
    # Compute predictions based on beta
    y_pred = (Z >= beta).astype(int)
    classification_calculator(y_pred,t_test)
    print(y_pred)




plt.plot(np.arange(iterations), BGD[0])
plt.show() 



#print(BGD)