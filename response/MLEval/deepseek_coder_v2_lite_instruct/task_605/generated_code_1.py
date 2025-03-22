import tensorflow as tf
import numpy as np
import json

def method():
    # Load the model
    # model = tf.keras.models.load_model('path_to_your_model.h5')
    
    # 修改为本地模型路径
    model = tf.keras.models.load_model('evaluation/dynamic_checking/baselines/MLEval/deepseek_coder_v2_lite_instruct/testcases/task_605_model_name.h5')

    # Load the history
    # with open('path_to_your_history.json', 'r') as file:
    # 修改为本地历史记录路径
    with open('evaluation/dynamic_checking/baselines/MLEval/deepseek_coder_v2_lite_instruct/testcases/task_605_history_file.json', 'r') as file:
        history = json.load(file)

    # Assuming the history contains loss and accuracy for training and validation
    loss = history['loss']
    accuracy = history['accuracy']

    # Example output: return the model and the history
    output = {
        'model': model,
        'history': history
    }

    return output

# Call the method for validation
result = method()
print(result)