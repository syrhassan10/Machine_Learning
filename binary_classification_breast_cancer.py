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


def sigmoid(x):
    return 1 / (1+ np.exp(-x))


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


def classification_calculator(y_true, y_pred,beta):
    # positive when t = 0
    # negative when t = 1

    if(y_true.shape != y_pred.shape):
        return 0
    
    print('**********************************************************************************', '\n')

    Neg = np.count_nonzero(y_true == 1)
    Pos = np.count_nonzero(y_true == 0)
  
    print('-------------------------------------- ', '\n')

    print('True: ', '\n')
    print('Negative',Neg, '\n')
    print('Positive', Pos, '\n')
  
  

    Neg_predict = np.count_nonzero(y_pred == 1)
    Pos_predict = np.count_nonzero(y_pred == 0)

    print('Prediction: ', '\n')
    print('Negative',Neg_predict, '\n')
    print('Positive',Pos_predict, '\n')

    false_pos =0
    false_neg =0
    true_neg =0
    true_pos =0


    for i in range(len(y_true)):
        if(y_true[i] == y_pred[i]):
            if(y_true[i] == 1):
                true_neg = true_neg + 1
            else:
                true_pos = true_pos + 1
        else:
            if(y_true[i] == 1):
                false_pos = false_pos + 1
            else:
                false_neg = false_neg + 1

    print('##################################', '\n')

    print('True Positive: ',true_pos ,'\n')
    print('True Negative: ',true_neg ,'\n')
    print('False Positive: ',false_pos ,'\n')
    print('False Negative: ',false_neg ,'\n')

    print('accuracy: ', 100*((true_pos + true_neg)/(Pos + Neg)), '% \n')
    print('##################################', '\n')
    
    print('**********************************************************************************', '\n')

    acc = 100*((true_pos + true_neg)/(Pos + Neg))
    if(true_pos == 0):
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    recall = true_pos / Pos

    fp_rate = false_pos / Neg
    
    tp_rate = true_pos / Pos
    #true positives : y =p, t=p; their number is denoted TP;
    #false positives : y =p, t=n; their number is denoted FP.
    #true negatives : y =n, t=n; their number is denoted TN
    #false negatives : y =n, t=p; their number is denoted FN.

    if(precision == 0 and recall == 0):
        f1_score = -1 # undefined
    else:
        f1_score = 2* ((precision*recall)/(precision+recall))

    misclassification_rate = 100 * (false_pos+false_neg)/(Pos+Neg) 
    return precision, recall, f1_score, misclassification_rate, acc,fp_rate, tp_rate

def plot_learning_rate_vs_loss(i1, i2, i3, BGD1,BGD2, BGD3,title):
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(i1), BGD1, label='Alpha: 0.0001')
    plt.plot(np.arange(i2), BGD2, label='Alpha: 0.01')
    plt.plot(np.arange(i3), BGD3, label='Alpha: 0.9')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def Threshold_finder(Z):

    np.sort(Z)
    accuracy_BGD = []
    precision_arr = []
    recall_arr = []
    f1_score_arr = []
    misclassification_rate_arr = []
    beta_arr = []

    fp_rate_arr =[]
    tp_rate_arr =[]

    for beta in Z:
        # Compute predictions based on beta
        y_pred = (Z >= beta).astype(int)
        precision, recall, f1_score, misclassification_rate, acc, fp_rate, tp_rate = classification_calculator(t_test, y_pred,beta)
        accuracy_BGD.append(acc)
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_score_arr.append(f1_score)
        misclassification_rate_arr.append(misclassification_rate)
        beta_arr.append(beta)
        fp_rate_arr.append(fp_rate)
        tp_rate_arr.append(tp_rate)
        #print(y_pred)


    min_x = 100

    # Loop through list to find max x value and its index
    i =0
    index = 0
    for misclas_rate in misclassification_rate_arr:
        if misclas_rate < min_x:
            min_x = misclas_rate
            max_beta = beta_arr[i]
            index = i
            
        i = i + 1
    print("Min Misclassification rate is: ", min_x)
    print("It occurs at beta:", max_beta)


    return precision_arr, recall_arr, f1_score_arr, misclassification_rate_arr, beta_arr, fp_rate_arr,tp_rate_arr,index

X1_train, t_train, X1_test, t_test = standardized_data_with_bias()

iterations = 1000
a1 = 0.0001
BGD1 = batch_gradient_descent(X1_train,t_train,a1,iterations)

iterations2 = 500
a2 = 0.01
BGD2 = batch_gradient_descent(X1_train,t_train,a2,iterations2)

iterations3 = 200
a3 = 0.5
BGD3 = batch_gradient_descent(X1_train,t_train,a3,iterations3)


plot_learning_rate_vs_loss(iterations, iterations2, iterations3, BGD1[0],BGD2[0], BGD3[0], 'Learning Rate Vs. Loss Function BGD')

# Using weights from BGD for test data
Z = np.dot(X1_test,BGD3[1])


precision_arr, recall_arr, f1_score_arr, misclassification_rate_arr, beta_arr,fp_rate_arr,tp_rate_arr,index=Threshold_finder(Z)


print('Misclassification Rate:',misclassification_rate_arr[index])
print('F1 Score:',f1_score_arr[index])

plt.scatter(recall_arr, precision_arr, label='Precision/Recall Curve', c='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision/Recall Curve')
plt.legend()
plt.grid(True)
plt.show() 


plt.figure(figsize=(10, 6))
plt.scatter(fp_rate_arr, tp_rate_arr, marker='o', c='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve BGD')
plt.legend()
plt.grid(True)
plt.show()
