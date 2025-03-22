import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=3, activation='relu')(gap)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)
    reshaped = Reshape(target_shape=(32, 32, 3))(dense2)
    element_wise_mul = Multiply()([reshaped, input_layer])

    flatten = Flatten()(element_wise_mul)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model