import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Second block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv4)

    # Main path
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    gap = GlobalAveragePooling2D()(conv6)

    # Channel weights
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Reshape weights
    reshape = Reshape((64, 1, 1))(fc2)

    # Multiply weights and input
    upsample = UpSampling2D(size=(4, 4))(reshape)
    output = Add()([upsample, conv6])

    # Classification layer
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model