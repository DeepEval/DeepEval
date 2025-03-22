import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path1: Two blocks of convolution followed by average pooling
    def block_path1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
        max_pooling2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return max_pooling2

    path1_output = block_path1(input_layer)
    path1_output = block_path1(path1_output)

    # Path2: Single convolutional layer
    path2_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the outputs from both pathways
    adding_layer = Add()([path1_output, path2_output])

    # Flatten the output
    flatten_layer = Flatten()(adding_layer)

    # Map the output to a probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model