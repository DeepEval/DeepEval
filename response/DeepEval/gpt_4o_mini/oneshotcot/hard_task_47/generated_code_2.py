import keras
from keras.layers import Input, Lambda, BatchNormalization, Concatenate, Dense, Flatten, AveragePooling2D, Conv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Depthwise Separable Convolutions
    def depthwise_separable_block(input_tensor):
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])

        merged = Concatenate()([path1, path2, path3])
        batch_norm = BatchNormalization()(merged)

        return batch_norm

    block1_output = depthwise_separable_block(input_layer)

    # Second Block: Multiple Branches for Feature Extraction
    def feature_extraction_block(input_tensor):
        path1 = Concatenate()([
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        ])

        path2 = Concatenate()([
            Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(input_tensor),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        ])

        path3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)

        merged = Concatenate()([path1, path2, path3])

        return merged

    block2_output = feature_extraction_block(block1_output)
    flatten_layer = Flatten()(block2_output)

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model