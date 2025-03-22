import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional block
    conv1 = Conv2D(32, (3, 3), activation='relu')

    # Define the second convolutional block
    conv2 = Conv2D(64, (3, 3), activation='relu')

    # Define the attention weights
    attn_weights = Conv2D(64, (1, 1), activation='softmax')

    # Define the attention layer
    attn_layer = Lambda(lambda x: attn_weights(x) * x)

    # Define the model
    model = keras.Sequential([
        # Input layer
        Input(shape=input_shape),

        # First convolutional block
        conv1,

        # Second convolutional block
        conv2,

        # Attention weights
        attn_layer,

        # Flatten layer
        Flatten(),

        # Fully connected layer
        Dense(128, activation='relu'),

        # Output layer
        Dense(10, activation='softmax')
    ])

    return model