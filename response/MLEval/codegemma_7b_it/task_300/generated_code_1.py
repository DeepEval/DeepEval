import pandas as pd
from sklearn.linear_model import LogisticRegression

def method():
    # Load the training data
    # train_data = pd.read_csv("path/to/training_data.csv")

    # 修改为本地数据文件
    train_data = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_300.csv")

    # Separate the features and labels
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Return the trained model
    return model

# Call the method to generate the trained model
trained_model = method()