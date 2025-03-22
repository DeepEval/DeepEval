import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First average pooling layer
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)

    # Second average pooling layer
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)

    # Third average pooling layer
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    # Concatenate the outputs of the average pooling layers
    concat_layer = Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_layer)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model