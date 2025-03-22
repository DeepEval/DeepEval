import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
# df = pd.read_csv("data.csv")

# 修改为本地数据文件
df = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_56.csv")

# Preprocess the data
df = df.drop(columns=["id"])  # Drop id column
df = pd.get_dummies(df, drop_first=True)  # One-hot encode categorical variables
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Generate the final output
output = {
    "model": clf,
    "accuracy": accuracy,
}

# Validate the output
print(output)