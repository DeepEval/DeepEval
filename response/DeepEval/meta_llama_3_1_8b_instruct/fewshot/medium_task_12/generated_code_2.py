import keras
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn2)
        bn3 = BatchNormalization()(conv3)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn2)
        bn3 = BatchNormalization()(conv3)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_3(input_tensor):
        conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        output_tensor = Concatenate()([conv1, conv2])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)

    flatten = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model