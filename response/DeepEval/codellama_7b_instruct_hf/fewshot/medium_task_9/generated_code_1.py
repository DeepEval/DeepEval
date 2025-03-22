import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Basic block
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn = BatchNormalization()(conv)
    relu = ReLU()(bn)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu)

    # Feature fusion
    adding_layer = Add()([maxpool, conv])

    # Second convolutional layer
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(adding_layer)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)

    # Final output layer
    flatten = Flatten()(relu2)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model