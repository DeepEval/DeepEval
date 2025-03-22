import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
# df = pd.read_csv('data.csv')

# 修改为本地数据文件
df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_165.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Initialize the KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
predictions = knn.predict(X_test)

# Calculate the accuracy score on the test data
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy score
print(f'Accuracy with 5 neighbors: {accuracy:.3f}')

# Now let's see what it's like when we use only 1 neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy with 1 neighbor: {accuracy:.3f}')

# Validation
output = accuracy
# return output