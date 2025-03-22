import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, concatenate

def dl_model():
    # Define the input layer
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # First convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    # First average pooling layer with 1x1 window
    x1 = AveragePooling2D(pool_size=(1, 1), strides=1)(x)

    # Second convolutional layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    # Second average pooling layer with 2x2 window
    x2 = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    # Third convolutional layer
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    # Third average pooling layer with 4x4 window
    x3 = AveragePooling2D(pool_size=(4, 4), strides=4)(x)

    # Concatenate the outputs of the pooling layers
    concatenated = concatenate([Flatten()(x1), Flatten()(x2), Flatten()(x3)])

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # First fully connected layer
    x = Dense(units=256, activation='relu')(flattened)
    # Second fully connected layer
    outputs = Dense(units=10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()