import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv("data.csv")

# Label encode the is_person column
le = LabelEncoder()
df["is_person"] = le.fit_transform(df["is_person"])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("is_person", axis=1), df["is_person"], test_size=0.2)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using accuracy score
accuracy = model.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Generate the output
output = {
    "y_pred": y_pred,
    "accuracy": accuracy
}

return output