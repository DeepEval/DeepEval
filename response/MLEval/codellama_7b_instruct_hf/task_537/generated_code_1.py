import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
y_pred_class = np.array([1 if y > 0.5 else 0 for y in y_pred])
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Accuracy: {accuracy:.2f}')

# Check if the model meets the requirement of accuracy > 85% of the testing set
if accuracy >= 0.85:
    output = 'The model meets the requirement of accuracy > 85% of the testing set.'
else:
    output = 'The model does not meet the requirement of accuracy > 85% of the testing set.'

# Return the output
return output