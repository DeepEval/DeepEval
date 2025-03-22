import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def method():
    # Load the dataset
    # df = pd.read_csv("your_data.csv")

    # 修改为本地数据文件
    df = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_431.csv")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF vectorizer and logistic regression
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Evaluate the performance of the model
    accuracy = pipeline.score(X_test, y_test)
    print("Accuracy:", accuracy)

    # Return the final output
    return pipeline

# Call the method for validation
output = method()