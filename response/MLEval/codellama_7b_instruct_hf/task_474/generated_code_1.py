import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Confusion matrix: {conf_mat}')
print(f'Accuracy: {accuracy:.3f}')

# Return the final output
return {'confusion_matrix': conf_mat, 'accuracy': accuracy}