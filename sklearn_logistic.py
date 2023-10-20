from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

# Train classifier
classifier = LogisticRegression(solver='lbfgs', max_iter=10000)
classifier.fit(X_train, y_train)

# Get predicted probabilities and predictions
y_score = classifier.predict_proba(X_test)[:, 1]  # Only take the positive class
y_pred = classifier.predict(X_test)

# Compute ROC curve and area
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

# Compute F1 Score
f1 = f1_score(y_test, y_pred)

# Compute Misclassification Rate
misclassification_rate = 1 - accuracy_score(y_test, y_pred)

# Plotting the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()

# Plotting the Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, color='b', lw=lw, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Example')
plt.legend(loc="upper right")
plt.show()

# Output F1 Score and Misclassification Rate
print("F1 Score: ", f1)
print("Misclassification Rate: ", misclassification_rate)
