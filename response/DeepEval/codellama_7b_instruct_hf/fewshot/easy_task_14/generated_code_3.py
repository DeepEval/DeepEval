import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model