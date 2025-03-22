import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'feature1': [0.5, 0.6, np.nan, 0.8, 0.9, 1.0, np.nan, 1.2],
    'feature2': [1.5, np.nan, 1.8, 1.9, 2.0, np.nan, 2.2, 2.3],
    'target': [0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

def method():
    # Dropping rows with any missing values
    df_non_missing = df.dropna()

    # Features and target
    X = df_non_missing[['feature1', 'feature2']]
    y = df_non_missing['target']

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and fitting the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Optionally return the fitted model or test set performance
    score = model.score(X_test, y_test)
    return score

# Calling the method for validation
output = method()
print("Model test set accuracy:", output)