import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    def block_path1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool1

    path1_output = block_path1(input_tensor=input_layer)
    path1_output = block_path1(input_tensor=path1_output)

    # Path 2: Single convolutional layer
    conv_path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool_path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_path2)

    # Addition of the two pathways
    added_output = Add()([path1_output, pool_path2])

    # Flatten the output
    flatten_layer = Flatten()(added_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model