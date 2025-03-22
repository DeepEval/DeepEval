import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def method():
    # Load a sample dataset
    data = datasets.load_iris()
    X = data.data[:, :2]  # Use only the first two features for visualization
    y = data.target

    # Only select two classes for binary classification
    X = X[y != 2]
    y = y[y != 2]

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM with a polynomial kernel of degree 2
    clf = SVC(kernel='poly', degree=2, C=1.0, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Retrieve support vectors
    support_vectors = clf.support_vectors_

    # Print the support vectors
    print("Support Vectors:")
    print(support_vectors)

    # Visualize the decision boundary and support vectors
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    # Create a grid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('SVM with Polynomial Kernel (degree=2)')
    plt.legend()
    plt.show()

    # Return the support vectors for any further validation or use
    return support_vectors

# Call the method function for validation
output = method()