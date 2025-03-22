import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Reshape, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Branch path
    avgpool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=16, activation='relu')(avgpool)
    dense2 = Dense(units=8, activation='relu')(dense1)
    reshaped = Reshape(target_shape=(4, 4))(dense2)

    # Merge paths
    concat = Concatenate()([pool1, reshaped])
    flatten = Flatten()(concat)

    # Output layers
    dense3 = Dense(units=32, activation='relu')(flatten)
    dense4 = Dense(units=16, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model