import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Define the feature extraction for each group
    def feature_extraction(input_tensor, kernel_size):
        return Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)

    # Apply feature extraction to each channel group
    conv1x1 = feature_extraction(split_layer[0], (1, 1))
    conv3x3 = feature_extraction(split_layer[1], (3, 3))
    conv5x5 = feature_extraction(split_layer[2], (5, 5))

    # Concatenate the outputs from the three groups
    concat_layer = Concatenate()([conv1x1, conv3x3, conv5x5])

    # Add batch normalization
    batch_norm = BatchNormalization()(concat_layer)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model