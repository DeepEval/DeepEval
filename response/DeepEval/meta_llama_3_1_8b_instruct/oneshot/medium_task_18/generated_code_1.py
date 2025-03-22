import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have size 32x32x3
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)  # 3x3 max pooling

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)  # 3x3 max pooling
        output_tensor = Concatenate()([conv1, conv2, conv3, pool])

        return output_tensor

    block_output = block(input_tensor=pool1)
    bath_norm = BatchNormalization()(block_output)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(bath_norm)  # 3x3 max pooling

    flatten_layer = Flatten()(pool2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model