import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import accuracy_score

def method(data, target_variable):
    """
    This function demonstrates a basic machine learning workflow. 

    Replace the example model and target variable with your specific needs.

    Args:
        data (pd.DataFrame): Your dataset.
        target_variable (str): The name of the column containing the target you want to predict.

    Returns:
        float: The accuracy score of the model.
    """

    # 1. Data Preprocessing (Example: Removing irrelevant columns)
    # ... (Code to select relevant features and handle missing data)

    # 2. Split Data into Training and Testing Sets
    X = data.drop(target_variable, axis=1)  # Features
    y = data[target_variable]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Choose and Train a Model (Example: Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Make Predictions
    y_pred = model.predict(X_test)

    # 5. Evaluate Model Performance
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Example Usage (Replace with your actual data and target variable)
# data = pd.read_csv("your_dataset.csv") 

# 修改为本地数据文件
data = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_220.csv")

target_variable = "your_target_column"  

result = method(data, target_variable)
print(f"Model Accuracy: {result}")