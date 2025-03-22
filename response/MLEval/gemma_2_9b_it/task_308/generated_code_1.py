import pandas as pd
from sklearn.linear_model import LinearRegression

def method():
    # Load your data into a pandas DataFrame. 
    # Assuming your DataFrame is named 'data' and has columns 'total_request_rate' and 'cpu'.
    # data = pd.read_csv('your_data.csv')  

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_308.csv')

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model to your data
    model.fit(data[['total_request_rate']], data['cpu'])

    # You can now use the model to make predictions.
    # For example, to predict the CPU usage for a given total request rate:
    # prediction = model.predict([[100]])  # Replace 100 with your desired request rate
    
    # Return the trained model
    return model

# Call the method and store the output
model = method()

# You can now use the 'model' object to make predictions 
# or save it for later use.