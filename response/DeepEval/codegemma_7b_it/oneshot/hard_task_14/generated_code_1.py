import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, Add, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main)
    avg_pool_main = GlobalAveragePooling2D()(max_pool_main)
    dense1_main = Dense(units=512, activation='relu')(avg_pool_main)
    dense2_main = Dense(units=input_layer.shape[3], activation='sigmoid')(dense1_main)
    reshape_main = Reshape((input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]))(dense2_main)
    multiply_main = Multiply()([reshape_main, input_layer])

    # Branch path
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)
    conv_branch = Conv2D(filters=input_layer.shape[3], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch)

    # Combine paths
    add_layer = Add()([multiply_main, conv_branch])

    # Fully connected layers
    flatten_layer = Flatten()(add_layer)
    dense1_shared = Dense(units=512, activation='relu')(flatten_layer)
    dense2_shared = Dense(units=256, activation='relu')(dense1_shared)
    dense3_shared = Dense(units=10, activation='softmax')(dense2_shared)

    model = keras.Model(inputs=input_layer, outputs=dense3_shared)

    return model