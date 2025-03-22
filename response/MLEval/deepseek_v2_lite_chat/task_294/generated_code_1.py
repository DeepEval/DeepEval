import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import statsmodels.api as sm


np.random.seed(0)
num_entries = 1000

# Age and positive nodes (fictional concept, lets say health points) are negatively correlated
age_min, age_max = 0, 100
positive_nodes_min, positive_nodes_max = 100, 200

# Operation year is a continuous variable
operation_year_min, operation_year_max = 1900, 2020
operation_year_step = 1

# Generate random data
data = {
    "Age": np.random.uniform(age_min, age_max, num_entries),
    "PositiveNodes": np.random.uniform(positive_nodes_min, positive_nodes_max,
  num_entries),
    "OperationYear": np.random.uniform(operation_year_min, operation_year_max,
  num_entries)
}

df = pd.DataFrame(data)

# Survival rate at 30% (0.3) for simplicity
survival_rate = 0.3

# Generate survival column based on the survival rate
df["Survived"] = np.random.choice([0, 1], size=num_entries, p=[survival_rate, 1-survival_rate])

def method():
    # Correlation matrix
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix)

    # Scatter plots for Age vs PositiveNodes, and Age vs OperationYear
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Age", y="PositiveNodes", hue="Survived", palette="coolwarm")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Age", y="OperationYear", hue="Survived", palette="coolwarm")
    plt.show()

    # Correlation with survival
    corr_survived = corr_matrix["Survived"].drop("Survived")
    print("\nCorrelation with Survival:")
    print(corr_survived)

    # Linear regression models
    X = df[["Age", "PositiveNodes", "OperationYear"]]
    y = df["Survived"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nMean Squared Error (MSE):", mse)
