import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Lambda, Multiply, add, Activation

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    avg_pool = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(units=32, activation='relu')(avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    reshape = Reshape((1, 1, 32))(dense2)

    mul = Multiply()([reshape, conv2])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(mul)

    output = add([conv3, input_layer])
    output = Activation('relu')(output)

    flatten_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model