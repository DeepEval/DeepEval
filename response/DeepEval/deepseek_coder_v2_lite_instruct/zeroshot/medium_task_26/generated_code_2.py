import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 64))

    # 1x1 convolutional layer to compress the input channels
    compressed = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Two parallel paths
    # Path 1: 1x1 convolutional layer to expand features back to original channels
    path1 = Conv2D(32, (1, 1), activation='relu')(compressed)

    # Path 2: 3x3 convolutional layer to expand features
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(compressed)

    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Two fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# model = dl_model()
# model.summary()