import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)
    inputs = Input(shape=input_shape)

    # Step 1: 1x1 Convolution to compress channels
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Step 2: Two parallel convolutional layers
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Step 3: Concatenate the results from the two parallel layers
    concatenated = Concatenate()([conv1, conv2])

    # Step 4: Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Step 5: Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()