import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Second block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(avg_pool)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
    gap = GlobalAveragePooling2D()(conv4)

    # Channel weights
    flatten_gap = Flatten()(gap)
    dense_weights = Dense(units=32, activation='relu')(flatten_gap)
    dense_weights = Dense(units=64, activation='relu')(dense_weights)

    # Multiply channel weights with input
    reshape_weights = Reshape((1, 1, 64))(dense_weights)
    multiply_output = multiply([gap, reshape_weights])

    # Concatenate main path and auxiliary path
    concat_layer = Concatenate()([multiply_output, avg_pool])

    # Output layer
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model