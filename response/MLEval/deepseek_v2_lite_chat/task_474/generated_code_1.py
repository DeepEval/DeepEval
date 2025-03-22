import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Load the dataset
    # Assuming you have a CSV file named 'data.csv' with columns 'Feature1', 'Feature2', etc.
    # data = pd.read_csv('data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_474.csv')

    # Assuming the target variable is 'Target'
    X = data[['Feature1', 'Feature2', 'Feature3']]  # Input features
    y = data['Target']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression object
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy score
    return accuracy

# Call the method and print the result
output = method()
print("Accuracy of the Logistic Regression Model:", output)