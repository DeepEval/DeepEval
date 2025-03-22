from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

def method():
    # Load your dataset
    # Replace 'data.csv' with your actual dataset path
    # data = pd.read_csv('data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_513.csv')
    
    # Assume that 'target' is the label column and 'features' are the predictor columns
    # Replace 'target' and 'features' with your actual column names
    target = data['target']
    features = data['features']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the accuracy of the model
    return accuracy

# Call the method for validation
output = method()
print("Model Accuracy:", output)