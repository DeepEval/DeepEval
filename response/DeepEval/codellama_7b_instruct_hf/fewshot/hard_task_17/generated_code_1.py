import keras
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Add, BatchNormalization, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Global Average Pooling
    gaps = GlobalAveragePooling2D()(input_layer)
    dense_1 = Dense(units=64, activation='relu')(gaps)
    reshape = Reshape((64, 1))(dense_1)
    fc_1 = Dense(units=32, activation='relu')(reshape)

    # Block 2: Convolutional Block
    conv_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_block)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    add = Add()([conv_2, fc_1])

    # Block 3: Fully Connected Block
    fc_2 = Dense(units=128, activation='relu')(add)
    fc_3 = Dense(units=64, activation='relu')(fc_2)
    output = Dense(units=10, activation='softmax')(fc_3)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model