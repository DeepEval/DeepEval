import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channels
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Define a function to create the feature extraction block for each channel
    def feature_extraction_block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return path1, path2, path3

    # Apply the feature extraction block to each channel
    path1, path2, path3 = feature_extraction_block(split_layer[0]), feature_extraction_block(split_layer[1]), feature_extraction_block(split_layer[2])

    # Concatenate the outputs from the three paths
    concatenated_output = Concatenate(axis=3)([path1, path2, path3])

    # Add batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(batch_norm)

    # Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model