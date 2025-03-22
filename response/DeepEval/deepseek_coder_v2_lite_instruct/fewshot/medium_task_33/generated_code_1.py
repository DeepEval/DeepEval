import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Lambda, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the feature extraction for each channel group
    def feature_extraction(input_tensor, kernel_size):
        return SeparableConv2D(filters=32, kernel_size=kernel_size, activation='relu')(input_tensor)

    # Apply feature extraction to each channel group
    conv1 = feature_extraction(split_layer[0], (1, 1))
    conv2 = feature_extraction(split_layer[1], (3, 3))
    conv3 = feature_extraction(split_layer[2], (5, 5))

    # Concatenate the outputs from the three groups
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Pass through fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model