import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Feature extraction for each channel group
    def feature_extraction(input_tensor, kernel_size):
        return SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)

    path1 = feature_extraction(split_layer[0], (1, 1))
    path2 = feature_extraction(split_layer[1], (3, 3))
    path3 = feature_extraction(split_layer[2], (5, 5))

    # Concatenate the outputs from the three groups
    concatenated = Concatenate()([path1, path2, path3])

    # Batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model