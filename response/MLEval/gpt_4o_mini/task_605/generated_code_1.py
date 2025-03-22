import tensorflow as tf
import pickle

def method():
    # Load the model
    # model_path = 'path/to/your/model.h5'  # Update with your model path

    # 修改为本地数据文件
    model_path = 'evaluation/dynamic_checking/baselines/MLEval/gpt_4o_mini/testcases/task_605_model_name.h5'
    
    model = tf.keras.models.load_model(model_path)

    # Load the training history
    # history_path = 'path/to/your/history.pkl'  # Update with your history path

    # 修改为本地数据文件
    history_path = 'evaluation/dynamic_checking/baselines/MLEval/gpt_4o_mini/testcases/task_605_training_history.pkl'

    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    # Output can include model summary and history
    output = {
        'model_summary': model.summary(),
        'history': history
    }
    
    return output

# Call the method for validation
output = method()
print(output)