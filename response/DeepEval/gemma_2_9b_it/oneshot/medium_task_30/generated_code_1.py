import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))

    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flattened1 = Flatten()(pool1)
    flattened2 = Flatten()(pool2)
    flattened3 = Flatten()(pool3)

    concatenated = Concatenate()([flattened1, flattened2, flattened3])

    flattened_concat = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened_concat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model