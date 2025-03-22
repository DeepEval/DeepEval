import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, multiply, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Branch path
    gap = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=128, activation='relu')(dense1)
    reshape = Reshape((128, 1, 1))(dense2)
    conv4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape)
    max_pooling2 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv4)

    # Add paths
    add = Add()([max_pooling, max_pooling2])
    flatten_layer = Flatten()(add)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model