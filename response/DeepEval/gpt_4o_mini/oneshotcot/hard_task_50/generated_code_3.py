import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Concatenate, Lambda, SeparableConv2D
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block: Three max pooling layers with different scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten each max pooling output
    flat1 = Flatten()(max_pool1)
    flat2 = Flatten()(max_pool2)
    flat3 = Flatten()(max_pool3)

    # Apply dropout to mitigate overfitting
    drop1 = Dropout(0.5)(flat1)
    drop2 = Dropout(0.5)(flat2)
    drop3 = Dropout(0.5)(flat3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(concatenated)

    # Reshape the output into a 4D tensor for the second block
    reshaped = Reshape((1, 1, 256))(dense1)  # Adjusting to (batch_size, 1, 1, 256)

    # Second block: Split the input into four groups using a Lambda layer
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each group with separable convolutions of varying kernel sizes
    conv_outputs = []
    for i, kernel_size in enumerate([1, 3, 5, 7]):
        conv_output = SeparableConv2D(filters=64, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_inputs[i])
        conv_outputs.append(conv_output)

    # Concatenate the outputs from the separable convolutions
    concatenated_output = Concatenate()(conv_outputs)

    # Flatten the final output before classification
    flatten_output = Flatten()(concatenated_output)

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model