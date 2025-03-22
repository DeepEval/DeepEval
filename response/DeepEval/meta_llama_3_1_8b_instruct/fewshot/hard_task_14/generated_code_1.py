import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    main_path = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=3*32*32, activation='relu')(main_path)
    weights = Reshape(target_shape=(32, 32, 3))(dense1)
    weights = Multiply()([weights, input_layer])

    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    adding_layer = Add()([weights, branch_path])
    flatten_layer = Flatten()(adding_layer)
    dense2 = Dense(units=128, activation='relu')(flatten_layer)
    dense3 = Dense(units=64, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model