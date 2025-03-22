import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First Block: Average Pooling
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool1 = Flatten()(pool1)

    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool2 = Flatten()(pool2)

    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    pool3 = Flatten()(pool3)

    # Concatenate the outputs of the pooling layers
    concat_output = Concatenate()([pool1, pool2, pool3])

    # Reshape the concatenated output to a 4-dimensional tensor
    reshape_output = Reshape(target_shape=(1, 1, 3))(concat_output)

    def block(input_tensor):

        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.2)(path1)

        # Path 2: Two 3x3 Convolutions
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(0.2)(path2)

        # Path 3: Single 3x3 Convolution
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(0.2)(path3)

        # Path 4: Average Pooling with 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.2)(path4)

        # Concatenate the outputs of the paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block_output = block(reshape_output)

    # Batch normalization and flatten
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)

    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model