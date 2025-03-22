import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    def path1_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(conv2)
        return avg_pool

    path1_output = path1_block(input_layer)
    path1_output = path1_block(path1_output)

    # Path 2: Single convolutional layer
    path2_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_output)
    path2_output = AveragePooling2D(pool_size=(2, 2), strides=2)(path2_output)

    # Combine outputs of both paths
    combined_output = Add()([path1_output, path2_output])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model