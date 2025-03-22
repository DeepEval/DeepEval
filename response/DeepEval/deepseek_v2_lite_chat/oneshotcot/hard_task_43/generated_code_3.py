import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, ZeroPadding2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv3)

    flat1 = Flatten()(avgpool3)
    concat = Concatenate()(flat1)

    dense1 = Dense(units=512, activation='relu')(concat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    dense3 = Dense(units=128, activation='relu')(dense2)

    # Reshape for second block
    reshape_layer = Flatten()(dense3)
    reshape_layer = Dense(units=(28*28*128))(reshape_layer)
    reshape_layer = keras.layers.Reshape((28, 28, 128))(reshape_layer)

    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)

    avgpool4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv6)

    concat2 = Concatenate()([conv4, conv5, conv6, avgpool4])

    dense4 = Dense(units=256, activation='relu')(concat2)
    dense5 = Dense(units=128, activation='relu')(dense4)
    output_layer = Dense(units=10, activation='softmax')(dense5)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model