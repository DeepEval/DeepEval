import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    flattened = Flatten()([flatten1, flatten2, flatten3])

    # Second block
    x = Flatten()(input_layer)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model