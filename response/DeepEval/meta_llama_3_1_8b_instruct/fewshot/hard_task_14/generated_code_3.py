import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Add, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    main_path = GlobalAveragePooling2D()(input_layer)
    weights = Dense(units=3, activation='linear')(main_path)
    reshaped_weights = Reshape(target_shape=(1, 1, 3))(weights)
    multiplied = Multiply()([input_layer, reshaped_weights])

    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    combined = Add()([multiplied, branch_path])

    x = Flatten()(combined)
    dense1 = Dense(units=64, activation='relu')(x)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model