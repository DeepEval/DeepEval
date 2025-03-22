import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=32, activation='relu')(pool1)
    weights = Dense(units=32, activation='softmax')(dense1)
    weights = Reshape((32, 1, 1))(weights)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = conv2

    output = Add()([branch_output, Multiply()([conv1, weights])])

    flatten_layer = Flatten()(output)
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model