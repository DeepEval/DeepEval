import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Create a simple Keras model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(10,)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Convert the Keras model to a TensorFlow Estimator
    estimator = tf.keras.estimator.model_to_estimator(model)

    # You can now use the estimator like any other TensorFlow Estimator
    # For example, to train the model on some data:
    # estimator.train(input_fn=your_input_fn)
    # or to make predictions:
    # predictions = estimator.predict(input_fn=your_input_fn)

    return estimator 

# Call the method and store the output
output = method() 

# Print the output (which is the TensorFlow Estimator)
print(output)