import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def conv_block(input_tensor, filters, kernel_size, dropout):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Dropout(rate=dropout)(conv)
        return conv

    conv1_1 = conv_block(input_tensor=input_layer, filters=32, kernel_size=(3, 3), dropout=0.2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = conv_block(input_tensor=pool1, filters=64, kernel_size=(3, 3), dropout=0.3)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)
    conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool2)

    conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    adding_layer = Add()([conv1_3, conv2_1])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model