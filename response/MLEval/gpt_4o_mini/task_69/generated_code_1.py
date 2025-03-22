import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_keras_model():
    # Create a simple Keras model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),  # Input layer with 32 features
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    return model

def method():
    # Create and compile the Keras model
    keras_model = create_keras_model()
    keras_model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])

    # Convert the Keras model to an Estimator
    estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model)

    return estimator

# Call the method for validation
output = method()
print(output)