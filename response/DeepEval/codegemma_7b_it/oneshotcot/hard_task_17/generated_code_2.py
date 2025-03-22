import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Weighted Feature Generation
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=64, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    reshape = Reshape((1, 1, 64))(dense2)

    # Block 2: Deep Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Block 2 Branch
    branch = GlobalAveragePooling2D()(max_pooling2)
    dense3 = Dense(units=64, activation='relu')(branch)
    reshape2 = Reshape((1, 1, 64))(dense3)

    # Fusion
    concat = Add()([reshape, reshape2])
    flatten_layer = Flatten()(concat)
    dense4 = Dense(units=128, activation='relu')(flatten_layer)
    dense5 = Dense(units=64, activation='relu')(dense4)
    output_layer = Dense(units=10, activation='softmax')(dense5)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model