import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def method():
    # Load the dataset
    # df = pd.read_csv("your_data.csv")

    # 修改为本地数据文件
    df = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_513.csv")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42)

    # Train the logistic regression model on the training set
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    return clf

# Call the method for validation
clf = method()