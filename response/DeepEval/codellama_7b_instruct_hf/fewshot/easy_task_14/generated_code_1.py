import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Reshape((32, 32, 1))(x)
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model