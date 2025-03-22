import keras
from keras.layers import Input, Conv2D, Add, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        branch_path = input_tensor
        adding_layer = Add()([conv2, branch_path])
        return adding_layer

    def block_2(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    output_layer = dense2

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model