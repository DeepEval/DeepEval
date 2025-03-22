import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def method():

    # Import data
    # data = pd.read_csv('data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_353.csv')

    X = data.drop('target', axis=1)
    y = data['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f'Accuracy: {accuracy}')

    # Get confidence scores
    confidence_scores = model.decision_function(X_test)

    # Find lowest confidence for correct output
    lowest_confidence_correct = confidence_scores[y_test == 1].min()

    # Find highest confidence for wrong output
    highest_confidence_wrong = confidence_scores[y_test != y_pred].max()

    # Print results
    print(f'Lowest confidence for correct output: {lowest_confidence_correct}')
    print(f'Highest confidence for wrong output: {highest_confidence_wrong}')

    return confidence_scores, y_test

# Call the method and get the confidence scores and ground truth
confidence_scores, y_test = method()

# Print the confidence scores
print(confidence_scores)

# Print the ground truth
print(y_test)