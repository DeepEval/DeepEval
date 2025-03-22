import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def method():
    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM classifier with a polynomial kernel of degree 2
    svm_poly = SVC(kernel='poly', degree=2)

    # Train the SVM classifier
    svm_poly.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_poly.predict(X_test)

    # Plot the decision boundary
    plot_decision_boundary(svm_poly, X, y)

    # Calculate and return the accuracy
    accuracy = svm_poly.score(X_test, y_test)
    output = f"Accuracy: {accuracy:.2f}"

    return output

def plot_decision_boundary(model, X, y):
    # Create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function values for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', label='Training data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary with Polynomial Kernel of Degree 2')
    plt.show()

# Call the method for validation
output = method()
print(output)