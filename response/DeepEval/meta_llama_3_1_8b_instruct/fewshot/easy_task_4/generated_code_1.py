import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Lambda

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
        output_tensor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return output_tensor

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
        output_tensor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        return output_tensor

    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    flatten = Flatten()(block2_output)

    dense1 = Dense(units=128, activation='relu')(flatten)
    add1 = Add()([dense1, flatten])
    dense2 = Dense(units=64, activation='relu')(add1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model