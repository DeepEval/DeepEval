from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, SeparableConv2D
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer for MNIST images (28x28, single channel)
    input_layer = Input(shape=(28, 28, 1))

    # First block with average pooling layers of different sizes
    pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flat_1x1 = Flatten()(pool_1x1)

    pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flat_2x2 = Flatten()(pool_2x2)

    pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat_4x4 = Flatten()(pool_4x4)

    # Concatenating the flattened outputs
    concatenated = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully connected layer followed by reshape
    dense_layer = Dense(units=1024, activation='relu')(concatenated)
    reshaped = Reshape((4, 8, 32))(dense_layer)  # Example reshape, adjust dimensions as needed

    # Second block: Split and process with separable convolutions
    def split_and_process(inputs):
        splits = tf.split(inputs, num_or_size_splits=4, axis=-1)
        
        conv1x1 = SeparableConv2D(32, (1, 1), activation='relu', padding='same')(splits[0])
        conv3x3 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(splits[1])
        conv5x5 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(splits[2])
        conv7x7 = SeparableConv2D(32, (7, 7), activation='relu', padding='same')(splits[3])

        return Concatenate()([conv1x1, conv3x3, conv5x5, conv7x7])

    processed = Lambda(split_and_process)(reshaped)

    # Final steps: flatten and fully connected layer for classification
    flattened = Flatten()(processed)
    output_layer = Dense(units=10, activation='softmax')(flattened)  # 10 output units for MNIST

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model