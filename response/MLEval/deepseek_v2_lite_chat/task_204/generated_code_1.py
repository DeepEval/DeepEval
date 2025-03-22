import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def method():
    # Load the iris dataset as an example
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Only take the first two features for visualization
    y = iris.target

    # Standardize the features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Create a copy of the original data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a SVM classifier with a polynomial kernel of degree 2
    svc = SVC(kernel='poly', degree=2)

    # Train the SVM
    svc.fit(X_train, y_train)

    # Predict on the test data
    y_pred = svc.predict(X_test)

    # Print accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the decision boundary
    # We'll use a dense grid of points for better visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('3-Class SVM with polynomial kernel of degree 2')

    plt.show()

    return y_pred

# Call the method for validation
output = method()