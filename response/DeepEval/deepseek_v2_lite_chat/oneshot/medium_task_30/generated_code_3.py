import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense
from keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))

    # Average Pooling Layers
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)

    # Flatten and Concatenate Layers
    flattened_1 = Flatten()(avg_pool_1)
    flattened_2 = Flatten()(avg_pool_2)
    flattened_3 = Flatten()(avg_pool_3)

    concatenated = Concatenate()([flattened_1, flattened_2, flattened_3])

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model