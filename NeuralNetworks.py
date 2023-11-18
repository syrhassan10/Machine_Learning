import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    # Lists to store features and labels
    features = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split line into components and convert them to float
            components = line.strip().split(',')
            if len(components) > 1:  # Check if line is not empty
                # All but the last element are features
                features.append([float(x) for x in components[:-1]])
                # The last element is the label
                labels.append(int(components[-1]))

    # Convert lists to numpy arrays for efficiency
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def data_setup(features, labels):

    # data splitting between train and test
    X_train, X_test, t_train, t_test = train_test_split(features, labels, test_size = 0.2, random_state =420)

    #feature standardization
    # Initialize the StandardScaler
    sc = StandardScaler()

    # Fit and transform the training data
    X_train = sc.fit_transform(X_train)

    # Transform the test data
    X_test = sc.transform(X_test)

    return X_train, t_train, X_test, t_test



file_path = 'data_banknote_authentication.txt'
training_data, target_data = load_data(file_path)

X_train, t_train, X_test, t_test = data_setup(training_data, target_data)


print(X_train.shape)
print(t_train.shape, '\n')

print('Test Set\n')

print(X_test.shape)
print(t_test.shape)

