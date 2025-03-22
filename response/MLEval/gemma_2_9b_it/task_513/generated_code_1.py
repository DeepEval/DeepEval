import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # **1. Load your dataset**
    # Replace 'your_dataset.csv' with the actual path to your dataset
    # data = pd.read_csv('your_dataset.csv') 

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_513.csv')

    # **2. Separate features (X) and target variable (y)**
    # Replace 'target_variable' with the name of your target column
    X = data.drop('target_variable', axis=1)
    y = data['target_variable']

    # **3. Split the data into training and testing sets**
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # **4. Create a Logistic Regression model**
    model = LogisticRegression()

    # **5. Train the model on the training data**
    model.fit(X_train, y_train)

    # **6. Make predictions on the test data**
    y_pred = model.predict(X_test)

    # **7. Evaluate the model's performance**
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # **8. Return the trained model (optional)**
    return model

# Call the method to train and evaluate the model
trained_model = method()