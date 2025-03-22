from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Branch path
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_1)

    # Combine outputs
    adding_layer = Add()([maxpool1, maxpool2, maxpool3])

    # Flatten and project onto a probability distribution
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model