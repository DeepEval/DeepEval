import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Lambda, SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Apply Dropout to mitigate overfitting
    drop1 = Dropout(0.5)(flat1)
    drop2 = Dropout(0.5)(flat2)
    drop3 = Dropout(0.5)(flat3)

    # Concatenate the flattened vectors
    concatenated = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer followed by reshape
    dense1 = Dense(units=1024, activation='relu')(concatenated)
    reshaped = Reshape((8, 8, 16))(dense1)  # Reshape to a 4D tensor

    # Second Block
    def split_and_process(input_tensor):
        # Split into four groups along the last dimension
        splits = tf.split(input_tensor, num_or_size_splits=4, axis=-1)

        # Apply SeparableConv2D with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(splits[3])

        return Concatenate()([conv1, conv2, conv3, conv4])

    processed_tensor = Lambda(split_and_process)(reshaped)

    # Flatten and final fully connected layer for classification
    flatten_layer = Flatten()(processed_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model