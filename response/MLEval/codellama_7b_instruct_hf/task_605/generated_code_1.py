import joblib

def method():
    # Load the trained model from a file
    # model = joblib.load("trained_model.pkl")

    # Load the training history from a file
    # history = joblib.load("training_history.pkl")

    # 修改为本地数据文件
    model = joblib.load("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_605_trained_model.pkl")
    history = joblib.load("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_605_training_history.pkl")

    # Print the model and training history
    print(model)
    print(history)

    # Return the model and training history
    return model, history

# Call the method for validation
model, history = method()