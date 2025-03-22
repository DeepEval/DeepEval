import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def method():

    # Load the dataset
    # data = pd.read_csv('cancer_dataset.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_294.csv')

    # Split the data into features and target variable
    features = data[['age', 'positive_nodes', 'operation_year']]
    target = data['survival']

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(features, target)

    # Predict the survival of new patients
    new_features = pd.DataFrame({
        'age': [50, 60, 70],
        'positive_nodes': [1, 2, 3],
        'operation_year': [2010, 2015, 2020]
    })
    predictions = model.predict(new_features)

    # Print the predictions
    print(predictions)

    return predictions

method()