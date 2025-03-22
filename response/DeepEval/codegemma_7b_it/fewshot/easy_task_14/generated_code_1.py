import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    gap = GlobalAveragePooling2D()(input_layer)

    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='sigmoid')(dense2)

    reshaped = Reshape(target_shape=(32, 32, 3))(dense3)
    output_layer = Dense(units=10, activation='softmax')(reshaped)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model