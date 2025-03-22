import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Branch Path
    branch_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=64, activation='relu')(branch_pool)
    dense2 = Dense(units=128, activation='relu')(dense1)
    channel_weights = Reshape(target_shape=(32, 32, 128))(dense2)
    weighted_input = Multiply()([input_layer, channel_weights])

    # Combination and Classification
    merged = Add()([pool, weighted_input])
    flatten = Flatten()(merged)
    dense3 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model