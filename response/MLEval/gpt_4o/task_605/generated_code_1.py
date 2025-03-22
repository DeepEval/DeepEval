import tensorflow as tf
import pickle

def method():
    # Load the saved Keras model
    model = tf.keras.models.load_model('evaluation/dynamic_checking/baselines/MLEval/gpt_4o/testcases/task_605_model_name.h5')

    # Load the training history from a pickle file
    with open('evaluation/dynamic_checking/baselines/MLEval/gpt_4o/testcases/task_605_training_history.pkl', 'rb') as file:
        history = pickle.load(file)

    # Create the output containing both model and history
    output = {
        'model': model,
        'history': history
    }

    return output

# Call the method for validation
output = method()

# Optional: Print model summary and history keys for validation
print(output['model'].summary())
print(output['history'].keys())