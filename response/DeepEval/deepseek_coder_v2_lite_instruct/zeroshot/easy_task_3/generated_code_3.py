import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # First convolutional block: <convolution, convolution, max pooling>
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolutional block: <convolution, convolution, max pooling>
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()