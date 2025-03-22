import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS

def method():

    # Load the dataset (replace with your actual data)
    # df = pd.read_csv('your_data.csv')

    # 修改为本地数据文件
    df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_533.csv')

    # Select the features and target variable
    X = df['feature_1']
    y = df['target_variable']

    # Create the regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X.values.reshape(-1, 1), y)

    # Remove outliers using the Tukey method
    Q1 = df['target_variable'].quantile(0.25)
    Q3 = df['target_variable'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    df_no_outliers = df[(df['target_variable'] >= lower_bound) & (df['target_variable'] <= upper_bound)]

    # Select the features and target variable for the reduced dataset
    X_no_outliers = df_no_outliers['feature_1']
    y_no_outliers = df_no_outliers['target_variable']

    # Create the regression model for the reduced dataset
    model_no_outliers = LinearRegression()

    # Fit the model to the reduced dataset
    model_no_outliers.fit(X_no_outliers.values.reshape(-1, 1), y_no_outliers)

    # Print the coefficients and intercept for both models
    print("Coefficients without outliers:", model_no_outliers.coef_)
    print("Intercept without outliers:", model_no_outliers.intercept_)
    print("Coefficients with outliers:", model.coef_)
    print("Intercept with outliers:", model.intercept_)

    # Return the output if needed
    return output

# Call the method for validation
method()