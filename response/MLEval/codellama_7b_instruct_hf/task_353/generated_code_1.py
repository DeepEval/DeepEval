import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def method():
    # Load the dataset
    # df = pd.read_csv('data.csv')

    # 修改为本地数据文件
    df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_353.csv')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['feature_1'], df['feature_2'], test_size=0.2)

    # Create a model
    model = RandomForestClassifier()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Show the lowest confidence for correct output
    incorrect_predictions = [i for i, p in enumerate(y_pred) if p != y_test[i]]
    lowest_confidence = [y_pred[i] for i in incorrect_predictions]
    print("Lowest confidence for correct output:", lowest_confidence)

    # Show the highest confidence for wrong output
    correct_predictions = [i for i, p in enumerate(y_pred) if p == y_test[i]]
    highest_confidence = [y_pred[i] for i in correct_predictions]
    print("Highest confidence for wrong output:", highest_confidence)

    # Keep the number of correct predictions at 95%
    num_correct_predictions = len([i for i, p in enumerate(y_pred) if p == y_test[i]])
    if num_correct_predictions < 0.95 * len(y_test):
        print("Number of correct predictions below 95%, adjusting model...")
        # Adjust the model to improve the accuracy
        # ...
    else:
        print("Number of correct predictions above 95%, model is accurate")

    return accuracy, precision, recall, f1

# Call the method for validation
accuracy, precision, recall, f1 = method()
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)