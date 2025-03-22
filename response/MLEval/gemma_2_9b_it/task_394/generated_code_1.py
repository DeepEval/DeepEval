import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # 1. Load your data
    # data = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with your actual file

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_394.csv')

    # 2. Preprocess your data (feature engineering, handling missing values, etc.)
    # Example:
    X = data[['feature1', 'feature2', 'feature3']]  # Select features
    y = data['target_variable']  # Select target variable

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Choose and initialize your model
    model = LogisticRegression() 

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 7. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # 8. Return the trained model (optional)
    return model

# Call the method to execute the code
trained_model = method()