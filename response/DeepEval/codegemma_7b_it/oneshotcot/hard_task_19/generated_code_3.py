import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Branch path
    gap = GlobalAveragePooling2D()(conv3)
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=128, activation='relu')(dense1)
    reshape = keras.layers.Reshape((128,))(dense2)
    dense3 = Dense(units=32, activation='relu')(reshape)

    # Combine paths
    output = Add()([max_pooling, dense3])

    # Classification layers
    flatten_layer = Flatten()(output)
    dense4 = Dense(units=10, activation='softmax')(flatten_layer)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=dense4)

    return model