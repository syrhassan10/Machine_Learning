import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def data_setup():

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    print(data.shape)

    # data splitting between train and test
    X_train, X_test, t_train, t_test = train_test_split(data, target, test_size = 0.2, random_state =420)

    #feature standardization
    # Initialize the StandardScaler
    sc = StandardScaler()

    # Fit and transform the training data
    X_train = sc.fit_transform(X_train)

    # Transform the test data
    X_test = sc.transform(X_test)



    print(X_train.shape)

    #print(X_test.shape)

    print(t_train.shape)
    #print(t_test.shape)

    return X_train, t_train, X_test, t_test

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def KNN_algo_cross_validation(X_train,t_train,k_num):

    cv_errors = np.zeros(k_num)  # Array to store the cross-validation errors for each k
    cv_errors_train= np.zeros(k_num)  # Array to store the cross-validation errors for each k
    
    set1_data = X_train[0:80]
    set1_target = t_train[0:80]

    set2_data = X_train[80:161]
    set2_target = t_train[80:161]

    set3_data = X_train[161:242]
    set3_target = t_train[161:242]

    set4_data = X_train[242:323]
    set4_target = t_train[242:323]

    set5_data = X_train[323:]
    set5_target = t_train[323:]

    sets_data = [set1_data,set2_data,set3_data,set4_data,set5_data]
    sets_target = [set1_target,set2_target,set3_target,set4_target,set5_target]
    
    '''
    fold_size = len(X_train) // 5
    cv_errors = np.zeros(k_num)

    # Create the 5 folds
    sets_data = [X_train[i*fold_size : (i+1)*fold_size] for i in range(5)]
    sets_target = [t_train[i*fold_size : (i+1)*fold_size] for i in range(5)]
    '''
    print(sets_data[0].shape)
    print(sets_data[1].shape)

    print(sets_data[2].shape)

    print(sets_data[3].shape)
    print(sets_data[4].shape)


    # Perform 5-fold cross-validation
    for k in range(1, k_num + 1):  # Start from 1 since k=0 is not valid
        fold_errors = []
        fold_errors_train = []
        
        for i in range(5):
            # Create training and validation sets for the i-th fold
            X_val, t_val = sets_data[i], sets_target[i]
            X_train_fold = np.concatenate(sets_data[:i] + sets_data[i+1:])
            t_train_fold = np.concatenate(sets_target[:i] + sets_target[i+1:])
            

            predictions = []
            predictions_train = []

            # For each point in the validation set
            for val_point in X_val:
                x=0
                distances = np.zeros(len(X_train_fold))
                for train_point in X_train_fold:
                    distances[x] = euclidean_distance(val_point, train_point)
                    x = x + 1
                
                nearest_neighbor_ids = distances.argsort()[:k]
                nearest_targets = t_train_fold[nearest_neighbor_ids]
                
                predicted_val = nearest_targets.mean()
                predictions.append(predicted_val)

            for train_point in X_train_fold:
                x=0
                distances = np.zeros(len(X_train_fold))
                for tp in X_train_fold:
                    distances[x] = euclidean_distance(train_point, tp)
                    x = x + 1
                
                nearest_neighbor_ids = distances.argsort()[:k]
                nearest_targets = t_train_fold[nearest_neighbor_ids]
                
                predicted_val = nearest_targets.mean()
                predictions_train.append(predicted_val)

            
            # calculate mean squared error for that fold
            fold_errors.append(mean_squared_error(t_val, predictions))
            fold_errors_train.append(mean_squared_error(t_train_fold, predictions_train))

        # average error across all folds for the current k
        cv_errors[k - 1] = np.mean(fold_errors)
        cv_errors_train[k - 1] = np.mean(fold_errors_train)

    k_arr = np.arange(1, k + 1)

    return cv_errors, cv_errors_train,k_arr

def plot_scatter(x,y,x2,y2,label, xlabel,ylabel,title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='red', label='Cross Validation Error')  # Scatter plot
    plt.scatter(x2, y2, c='purple', label='Training Error')  # Scatter plot
    plt.plot(x, y, c='red', label='Line')  # Line plot connecting the points
    plt.plot(x2, y2, c='blue', label='Line')  # Line plot connecting the points

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def KNN(X_train,t_train,X_test,t_test,k):
    predictions = []
    
    for test_point in X_test:
        x=0
        distances = np.zeros(len(X_train))
        for train_point in X_train:
            distances[x] = euclidean_distance(test_point, train_point)
            x = x + 1
        
        nearest_neighbor_ids = distances.argsort()[:k]
        nearest_targets = t_train[nearest_neighbor_ids]
        
        predicted_val = nearest_targets.mean()
        predictions.append(predicted_val)
    
    mean_squared_err = mean_squared_error(t_test, predictions)

    
    return mean_squared_err


X_train, t_train, X_test, t_test = data_setup()

errors,errors_train,ks = KNN_algo_cross_validation(X_train,t_train,10)
print(errors)

minIndex = np.argmin(errors)

plot_scatter(ks,errors,ks,errors_train, 'Er', 'K','Error', 'KNN Learning Training Eror vs 5 fold Cross Validtion ')

print('Lowest Cross Validation Error is ', errors[minIndex], 'which happens at K=: ', ks[minIndex])

print(KNN(X_train,t_train,X_test,t_test,ks[minIndex]))

