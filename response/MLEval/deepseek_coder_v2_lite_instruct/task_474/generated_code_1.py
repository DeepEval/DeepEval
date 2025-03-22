import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Assuming we have a dataset 'data' with features and target variable 'target'
    # data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your dataset file
    
    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_coder_v2_lite_instruct/testcases/task_474.csv')

    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Prepare the output
    output = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }

    return output

# Call the method for validation
result = method()
print(result)