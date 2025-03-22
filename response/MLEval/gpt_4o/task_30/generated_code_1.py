# Import necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method(input_shape=(28, 28), num_classes=10, hidden_layers=[128, 64], activation='relu'):
    """
    Builds a Deep Neural Network (DNN) model.
    
    Parameters:
    - input_shape: tuple, shape of the input data (excluding batch size).
    - num_classes: int, number of output classes.
    - hidden_layers: list, number of neurons in each hidden layer.
    - activation: str, activation function for hidden layers.

    Returns:
    - model: a compiled Keras model.
    """
    
    # Flatten the input shape for the first layer if needed
    input_shape_flattened = (input_shape[0] * input_shape[1],) if len(input_shape) > 1 else input_shape
    
    # Define the model
    model = Sequential()
    
    # Add the input layer
    model.add(Dense(hidden_layers[0], input_shape=input_shape_flattened, activation=activation))

    # Add hidden layers
    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, activation=activation))
    
    # Add the output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Call the method for validation
if __name__ == "__main__":
    dnn_model = method(input_shape=(28, 28), num_classes=10, hidden_layers=[128, 64])
    dnn_model.summary()