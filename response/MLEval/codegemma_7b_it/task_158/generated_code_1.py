from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ENCODING_DIM = 20  # Define the encoding dimension

def method():
    # Create a Keras Sequential model
    encoder = Sequential([
        Dense(256, activation='relu', input_shape=(784,)),  # Flatten the input image
        Dense(ENCODING_DIM)  # Output vector of size ENCODING_DIM
    ])

    # Return the encoder model
    return encoder

# Call the generated method for validation
encoder = method()
print(encoder.summary())