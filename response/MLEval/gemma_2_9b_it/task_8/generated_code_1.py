import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Load your dataset 
    # Replace 'your_dataset.csv' with the actual path to your dataset
    # data = pd.read_csv('your_dataset.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_8.csv')

    # Separate features (X) and target variable (y)
    X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your target column name
    y = data['target_variable']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Return the trained model (optional)
    return model

# Call the method to train and evaluate the model
trained_model = method()