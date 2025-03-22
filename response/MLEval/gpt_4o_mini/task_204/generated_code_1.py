import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def method():
    # Generate synthetic data
    X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the SVM model with a polynomial kernel of degree 2
    model = SVC(kernel='poly', degree=2, coef0=1, C=1)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the SVM with Polynomial Kernel (degree 2): {accuracy:.2f}')

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap='autumn', label='Train data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='winter', label='Test data', edgecolors='k')

    # Create a grid to plot decision boundaries
    xlim = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    ylim = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(xlim, ylim)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.title('SVM with Polynomial Kernel (degree 2)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    return accuracy

# Call the method for validation
output = method()