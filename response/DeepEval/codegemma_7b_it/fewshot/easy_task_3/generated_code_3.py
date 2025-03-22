import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LeakyReLU, Dropout, Conv2DTranspose, concatenate, Add

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return maxpool

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return maxpool

    def block_3(input_tensor):
        conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return maxpool

    def block_4(input_tensor):
        conv1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return maxpool

    def block_5(input_tensor):
        conv1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return maxpool

    block1_output = block_1(input_tensor)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)
    block4_output = block_4(block3_output)
    block5_output = block_5(block4_output)

    flatten = Flatten()(block5_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model