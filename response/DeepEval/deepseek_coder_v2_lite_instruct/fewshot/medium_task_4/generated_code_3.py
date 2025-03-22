import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool1

    def block_2(input_tensor):
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return pool2

    path1_output = block_1(input_tensor=input_layer)
    path1_output = block_2(input_tensor=path1_output)

    # Path 2
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    # Combine Paths
    adding_layer = Add()([path1_output, path2_output])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model