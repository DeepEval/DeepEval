import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    global_average_pooling = GlobalAveragePooling2D()(input_layer)
    dense_layer_1 = Dense(units=128, activation='relu')(global_average_pooling)
    dense_layer_2 = Dense(units=64, activation='relu')(dense_layer_1)
    reshape_layer = Flatten()(dense_layer_2)
    output_layer = Dense(units=10, activation='softmax')(reshape_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model