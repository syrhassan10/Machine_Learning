# -*- coding: utf-8 -*-
"""mahdih2_400240420_A1_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13i2EwHnH1OVspodpJNgZXKTvbCWZwwoE
"""

# -*- coding: utf-8 -*-
"""4Sl4 Assigment 1.ipynb
**Generation of Training Set**
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def create_data_set(seed, n_samples, variance):
  # Set random seed for reproducibility - Last 4 digits of my student id: 400240420
  np.random.seed(seed)

  # Generate 120 examples with features uniformly spaced in [0, 1]
  training_x = np.linspace(0, 1, n_samples)


  # Generate targets using the relation t = sin(4πx + π/2) + ε
  # where ε is a random noise
  std_dev = np.sqrt(variance)
  noise = np.random.normal(0, std_dev, n_samples)

  t = np.sin(4 * np.pi * training_x + np.pi / 2) + noise


  return training_x, t



def plot_data_sets(x_train, t_train, x_valid, t_valid,x_predict, y_predict, x_true, y_true):
  # Plot the generated validation set
  plt.figure(figsize=(10, 6))
  plt.scatter(x_train, t_train, label='Training Set (t) with noise ε', c='blue')
  plt.scatter(x_valid, t_valid, label='Validation Set (t) with noise ε', c='green')

  plt.plot(x_predict, y_predict, label='Prediction Function', c='purple')
  plt.plot(x_true, y_true, label='True Function (sin(4πx + π/2))', c='red')

  plt.xlabel('x')
  plt.ylabel('t')
  plt.title('Prediction')
  plt.legend()
  plt.grid(True)
  plt.show()


def build_up_matrix_model_complexity(m, X):
  final_x = []

  for i in range(m+1):
    tmp_0 = np.power(X,i)
    if i==0:
      final_x = tmp_0
    else:
      final_x = np.column_stack((final_x,tmp_0))

  return final_x


def train_model(x,t_train):
    # training the model based on training data
  if(m == 0):
    A = np.dot(x.T,x)
    A1=np.linalg.inv(A)

    t1 = np.dot(x.T,t_train)

    w = np.dot(A1,t1)
  else:
    A = np.dot(x.T,x)
    A1=np.linalg.inv(A)

    t1 = np.dot(x.T,t_train)

    w = np.dot(A1,t1)

  y_train_predict = np.dot(x,w)

  return y_train_predict, w



def train_model_regularization(x,t_train):
  N, D = x.shape

  smallest_error =100000000

  lambda_ = 0.000001
  while lambda_ <= 0.5:
    lambda_ += 0.000001
    B = np.zeros((D, D))
    np.fill_diagonal(B[1:, 1:], 2 * lambda_)
    w = np.linalg.inv(x.T @ x + (N/2) * B) @ x.T @ t_train
    y_train_regularized = np.dot(x,w)
    error  = calc_error(y_train_regularized, t_train)
    if(error <  smallest_error):
      smallest_error = error
      smallest_lambda = lambda_

  B = np.zeros((D, D))
  np.fill_diagonal(B[1:, 1:], 2 * smallest_lambda)
  w = np.linalg.inv(x.T @ x + (N/2) * B) @ x.T @ t_train
  y_train_regularized = np.dot(x,w)


  return y_train_regularized, w, smallest_lambda




def calc_error(y_predict, t_train):
  N = t_train.shape[0]
  #training error
  diff = np.subtract(t_train, y_predict)
  err = np.dot(diff.T,diff)/N
  return err





x_train, t_train = create_data_set(420,12,0.0625)
x_valid, t_valid = create_data_set(420,120,0.0625)


x_train = x_train.reshape(-1, 1)
x_valid = x_valid.reshape(-1, 1)


M_values = []
validation_errors = []
training_errors = []

for m in range(11):
  M_values.append(m)
  x_train_transformed = build_up_matrix_model_complexity(m,x_train)
  x_valid_transformed = build_up_matrix_model_complexity(m,x_valid)

  y_train, w = train_model(x_train_transformed, t_train)


  train_error = calc_error(y_train, t_train)
  training_errors.append(train_error)


  y_validation = np.dot(x_valid_transformed,w)
  valid_error = calc_error(y_validation, t_valid)
  validation_errors.append(valid_error)

  y_predict  = np.dot(x_valid_transformed, w)

  plot_data_sets(x_train, t_train, x_valid, t_valid, x_valid, y_predict, x_valid, np.sin(4 * np.pi * x_valid + np.pi / 2))




#plotting errors:

  # Generate 120 examples with features uniformly spaced in [0, 1]
x_true = np.linspace(0, 1, 120)

t_true = np.sin(4 * np.pi * x_true + np.pi / 2)

ase_true_valid_diff = np.subtract(t_valid, t_true)
ase_true_valid = np.dot(ase_true_valid_diff.T,ase_true_valid_diff)/120


#M = 11 Regularization
sc = StandardScaler()

XX_train = x_train
XX_valid = x_valid

XX_train = sc.fit_transform(XX_train)
XX_valid = sc.transform(XX_valid)
print(XX_train)

XX_train_A = build_up_matrix_model_complexity(m,XX_train)
XX_valid_A = build_up_matrix_model_complexity(m,XX_valid)


y_train, m11_w, lambda_chosen = train_model_regularization(XX_train_A, t_train)

y_valid_regularized = np.dot(XX_valid_A,m11_w)


print(lambda_chosen)

plot_data_sets(x_train, t_train, x_valid, t_valid, x_valid, y_valid_regularized, x_valid, np.sin(4 * np.pi * x_valid + np.pi / 2))


error_train  = calc_error(y_train, t_train)

error_valid  = calc_error(y_valid_regularized, t_valid)
M_values.append(11)
training_errors.append(error_train)

validation_errors.append(error_valid)



# Second Plot: Training and Validation Errors
plt.figure(figsize=(10, 6))
plt.scatter(M_values, training_errors, label='Training Errors', marker='o', c='blue')
plt.scatter(M_values, validation_errors, label='Validation Errors', marker='x', c='green')
plt.axhline(y=ase_true_valid, color='r', linestyle='--', label='Avg Squared Error of True Function')
plt.xlabel('M (Model Complexity)')
plt.ylabel('Error')
plt.title('Training and Validation Errors vs Model Complexity')
plt.legend()
plt.grid(True)
plt.show()

