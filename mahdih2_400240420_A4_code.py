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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_setup(features, labels, test_size=0.1, validation_size=0.2, random_state=420):
    """
    Splits the data into training, validation, and test sets, and standardizes the features.

    :param features: The feature data.
    :param labels: The labels.
    :param test_size: The proportion of the dataset to include in the test split.
    :param validation_size: The proportion of the training dataset to include in the validation split.
    :param random_state: The random state for reproducibility.
    :return: Tuple containing training, validation, and test sets for features and labels.
    """
    # Split the data into training + validation and test sets
    X_train_val, X_test, t_train_val, t_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Further split the training data into training and validation sets
    X_train, X_val, t_train, t_val = train_test_split(X_train_val, t_train_val, test_size=validation_size, random_state=random_state)

    # Initialize the StandardScaler
    sc = StandardScaler()

    # Fit and transform the training data
    X_train = sc.fit_transform(X_train)

    # Apply the same transformation to the validation and test data
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    return X_train, t_train, X_val, t_val, X_test, t_test

def sigmoid_activation(z):
    return 1 / (1+ np.exp(-z))



def sigmoid_derivative(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

# Initialize network parameters
def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    parameters = {
        "W1": np.random.randn(hidden_size1, input_size) ,
        "b1": np.zeros((hidden_size1, 1)),
        "W2": np.random.randn(hidden_size2, hidden_size1) ,
        "b2": np.zeros((hidden_size2, 1)),
        "W3": np.random.randn(output_size, hidden_size2) ,
        "b3": np.zeros((output_size, 1))
    }
    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    W1, b1, W2, b2, W3, b3 = parameters.values()
    
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid_activation(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid_activation(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid_activation(Z3)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache

# Backward propagation
def backward_propagation(X, Y, cache, parameters):
    m = X.shape[1]
    W2, W3 = parameters["W2"], parameters["W3"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m


    dZ2 = np.dot(W3.T, dZ3) * sigmoid_derivative(cache["Z2"])
    dW2 = np.dot(dZ2, A1.T) / m # taking the average of gradients in this layer
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m


    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(cache["Z1"])
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradients

# Update parameters
def update_parameters(parameters, gradients, learning_rate):
    for key in parameters:
        parameters[key] -= learning_rate * gradients["d" + key]
    return parameters

def compute_loss(A, Y):
    return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))

def validate(X_val, t_val, parameters):
    A, _ = forward_propagation(X_val, parameters)
    loss = compute_loss(A, t_val)
    return loss

def train(X_train, t_train, X_val, t_val, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, patience):
    parameters = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)
    best_loss = float('inf')
    no_improve_epoch = 0
    validation_loss_arr = []
    training_loss_arr = []

    for i in range(epochs):
        A, _ = forward_propagation(X_train, parameters)
        gradients = backward_propagation(X_train, t_train, _, parameters)
        parameters = update_parameters(parameters, gradients, learning_rate)

        # Validation
        val_loss = validate(X_val, t_val, parameters)
        train_loss = validate(X_train, t_train, parameters)

        validation_loss_arr.append(val_loss)
        training_loss_arr.append(train_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_parameters = parameters
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        if no_improve_epoch > patience:
            #print(f"Early stopping at epoch {i}")
            #print("best_loss", best_loss)
            break

        if i % 1000 == 0 or i == epochs - 1:
            #print(f"Epoch {i}: Training loss: {compute_loss(A, t_train)}, Validation loss: {val_loss}")
            print("best_loss", best_loss)


    return best_parameters, validation_loss_arr,training_loss_arr,i


def predict(X, parameters):
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5).astype(int)  # Convert boolean to int (0 or 1)
    return predictions

def misclassification_rate(predictions, true_labels):
    # Count the number of misclassified instances
    misclassified = np.sum(predictions != true_labels)
    
    # Calculate the misclassification rate
    total = len(true_labels)
    misclassification_rate = misclassified / total

    return misclassification_rate


def plot_scatter(x,y,x2,y2,label, xlabel,ylabel,title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, c='red', label='training Loss')  # Scatter plot
    plt.plot(x2, y2, c='blue', label='Validation Error')  # Scatter plot

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



file_path = 'data_banknote_authentication.txt'
training_data, target_data = load_data(file_path)

X_train, t_train, X_val, t_val, X_test, t_test = data_setup(training_data, target_data)



configurations = [(2, 6), (3, 5), (4, 4), (5, 3)]

for config in configurations:
    n1, n2 = config
    print(f"Training model with configuration: n1={n1}, n2={n2}")

    # Train the model
    params, validation_errors, training_errors, epoch_s = train(X_train, t_train, X_val, t_val, X_train.shape[1], n1, n2, 1, 0.005, 500, 5)

    # Generate epochs array for plotting
    epochs = np.arange(epoch_s + 1)

    # Plot training and validation errors
    #plot_scatter(epochs, training_errors, epochs, validation_errors, 's', 'Epochs', 'Cross-Entropy Loss', f'NeuralNet Configuration: n1={n1}, n2={n2}')

    # Make predictions on the test set
    predictions_test = validate(X_test,t_test,params)
    predictions_train = validate(X_train,t_train,params)
    predictions_validate = validate(X_val,t_val,params)


    print(predictions_test,predictions_train,predictions_validate)


    predict_classific_test = predict(X_test,params)
    predict_classific_train = predict(X_train,params)
    predict_classific_validate = predict(X_val,params)

    # Calculate misclassification rate
    misclassification_test = misclassification_rate(predict_classific_test, t_test)
    misclassification_train = misclassification_rate(predict_classific_train, t_train)
    misclassification_validate = misclassification_rate(predict_classific_validate, t_val)
    print(params)
    print(f"Misclassification Rate (TEST) for configuration n1={n1}, n2={n2}: {misclassification_test}\n")
    print(f"Misclassification Rate (TRAIN) for configuration n1={n1}, n2={n2}: {misclassification_train}\n")
    print(f"Misclassification Rate (Validate) for configuration n1={n1}, n2={n2}: {misclassification_validate}\n")

