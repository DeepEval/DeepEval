from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'data' and 'target' are already defined and contain the data and target values
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

# Initialize the KNN classifier with k=30
knn = KNeighborsClassifier(n_neighbors=30)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Return the accuracy as the output
return accuracy