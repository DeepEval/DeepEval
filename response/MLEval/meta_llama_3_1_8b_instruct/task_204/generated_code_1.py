# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Standardize features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Create a Support Vector Machine classifier
svm_poly = svm.SVC(kernel='poly', degree=2, C=1.0)
svm_poly.fit(X_train_std, y_train)

# Predict the response for test dataset
y_pred = svm_poly.predict(X_test_std)

# Print the accuracy of the classifier
print("Accuracy:", svm_poly.score(X_test_std, y_test))

# Visualize the decision boundary
x_min, x_max = X_train_std[:, 0].min(), X_train_std[:, 0].max()
y_min, y_max = X_train_std[:, 1].min(), X_train_std[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = svm_poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_std[:, 0], X_train_std[:, 1], c=y_train)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Polynomial SVM')
plt.show()

def method():
    return "Polynomial SVM with degree 2 is a quality model."

print(method())