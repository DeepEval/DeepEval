import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def multi_branch_conv(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

        conv4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)

        output_tensor = Concatenate()([max_pool1, max_pool2, max_pool3])

        return output_tensor

    multi_branch_output = multi_branch_conv(input_layer)

    flatten_layer = Flatten()(multi_branch_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model