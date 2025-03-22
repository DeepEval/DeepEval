import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

def method():
    # Define the encoding dimension
    ENCODING_DIM = 32  # or any other value you want for the encoding size
    
    # Define the input dimension for a flattened MNIST image
    INPUT_DIM = 28 * 28  # MNIST images are 28x28 pixels
    
    # Create a Sequential model
    encoder = Sequential([
        InputLayer(input_shape=(INPUT_DIM,)),  # specify the input shape
        Dense(ENCODING_DIM, activation='relu')  # create a dense layer with encoding dimension
    ])
    
    # We do not compile the model as per the task requirement
    
    # The output can be the encoder model itself
    output = encoder
    return output

# Call the method for validation
encoder_model = method()

# Display the summary of the encoder model to validate its structure
encoder_model.summary()