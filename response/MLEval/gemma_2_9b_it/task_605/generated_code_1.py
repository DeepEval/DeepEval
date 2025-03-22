import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History

def method():
    # Replace 'model_name.h5' with the actual filename of your saved model
    # model = load_model('model_name.h5') 

    # 修改为本地数据文件
    model = load_model('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_605_model_name.h5')

    # Replace 'history_file.json' with the actual filename of your history file
    # with open('history_file.json', 'r') as f:
    # 修改为本地数据文件
    with open('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_605_history_file.json', 'r') as f:
        history = History.from_json(f.read())

    return model, history

# Call the method to load the model and history
model, history = method() 

# Now you can use the loaded model and history for further tasks,
# such as prediction or evaluation.