import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    pool = GlobalAveragePooling2D()(input_layer)
    dense = Dense(input_shape=(32, 32, 3))(pool)
    dense = Dense(input_shape=(32, 32, 3))(dense)
    reshape = Reshape((32, 32, 3))(dense)
    output = keras.layers.Multiply()([input_layer, reshape])
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model