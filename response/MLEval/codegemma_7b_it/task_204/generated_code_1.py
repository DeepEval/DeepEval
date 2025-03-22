import numpy as np
from sklearn import svm

def method():
    # Generate sample data
    x = np.array([[0,0],[2,2],[3,3],[4,4],[5,5],[10,10],[11,11],[12,12],[13,13],[14,14]])
    y = np.array([0,0,0,0,0,1,1,1,1,1])

    # Create a polynomial kernel SVM model with degree 2
    clf = svm.SVC(kernel="poly", degree=2)

    # Fit the model to the data
    clf.fit(x, y)

    # Predict the labels for the data points
    y_pred = clf.predict(x)

    # Print the predicted labels
    print("Predicted labels:", y_pred)

    # Print the accuracy of the model
    print("Accuracy:", clf.score(x, y))

    # Return the predicted labels if needed
    return y_pred

# Call the generated method for validation
method()