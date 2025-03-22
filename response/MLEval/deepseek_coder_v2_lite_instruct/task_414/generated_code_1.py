import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # List to store the accuracy scores for different values of k
    accuracy_scores = []

    # Loop over different values of k
    for k in range(1, 31):
        # Create a KNN classifier with the current k value
        knn = KNeighborsClassifier(n_neighbors=k)
        # Train the classifier on the training data
        knn.fit(X_train, y_train)
        # Predict the test set labels
        y_pred = knn.predict(X_test)
        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        # Append the accuracy score to the list
        accuracy_scores.append(accuracy)

    # Plot the elbow curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), accuracy_scores, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Elbow Curve for K-Nearest Neighbors')
    plt.show()

    # Determine the best value of k based on the elbow curve
    best_k = np.argmax(accuracy_scores) + 1  # +1 because indices start at 0

    # Return the best value of k and the final output
    output = {
        'best_k': best_k,
        'accuracy_scores': accuracy_scores
    }

    return output

# Call the method for validation
output = method()
print(output)