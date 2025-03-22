import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply

def dl_model():  

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Branch Path
    gap = GlobalAveragePooling2D()(conv3)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    channel_weights = Reshape((32, 1))(dense2)

    # Weighted Sum
    weighted_input = Multiply()([conv3, channel_weights])

    # Concatenate
    combined_features = keras.layers.concatenate([pool, weighted_input], axis=3)

    # Final Classification Layers
    flatten = Flatten()(combined_features)
    dense3 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model