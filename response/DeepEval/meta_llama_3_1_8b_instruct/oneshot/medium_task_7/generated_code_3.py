import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Activation, MaxPooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # 32x32x3 images for CIFAR-10 dataset
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    def conv_block(input_tensor):
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = Add()([conv2, conv3])
        return output_tensor

    conv_block_output = conv_block(max_pooling1)

    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block_output)
    add_output = Add()([conv_block_output, conv4])

    bath_norm = BatchNormalization()(add_output)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(bath_norm)

    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    flatten_layer = Flatten()(conv5)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model