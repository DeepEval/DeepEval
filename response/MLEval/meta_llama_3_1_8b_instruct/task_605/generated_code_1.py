# Import necessary packages
import pickle
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the method function
def method():
    """
    Load model and history.

    Returns:
    -------
    output : dict
        A dictionary containing the loaded model and history.
    """

    # Load the model from the file
    # model = load_model('model.h5')

    # 修改为本地模型路径
    model = load_model('evaluation/dynamic_checking/baselines/MLEval/meta_llama_3_1_8b_instruct/testcases/task_605_model_name.h5')

    # Load the history from the file
    # with open('history.pkl', 'rb') as f:
    # 修改为本地数据文件
    with open('evaluation/dynamic_checking/baselines/MLEval/meta_llama_3_1_8b_instruct/testcases/task_605_training_history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Return the output as a dictionary
    return {
       'model': model,
        'history': history
    }

# Call the method for validation
output = method()
print(output)

# Example usage:
# To access the model and history from the output dictionary
loaded_model = output['model']
loaded_history = output['history']