import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.layers import Conv2D, MaxPooling2D, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    reshaped_weights = Reshape((1, 1, 32))(dense2)
    weighted_output = Multiply()([input_layer, reshaped_weights])

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch from Block 1 to Block 2
    branch_output = Add()([max_pool, weighted_output])

    # Classification layers
    flatten_layer = Flatten()(branch_output)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    dense4 = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=dense4)

    return model