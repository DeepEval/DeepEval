import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Convolution, Convolution, Max Pooling
    conv1a = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    act1a = Activation('relu')(conv1a)
    conv1b = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(act1a)
    act1b = Activation('relu')(conv1b)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(act1b)

    # Second block: Convolution, Convolution, Max Pooling
    conv2a = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool1)
    act2a = Activation('relu')(conv2a)
    conv2b = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2a)
    act2b = Activation('relu')(conv2b)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(act2b)

    # Third block: Convolution, Convolution, Convolution, Max Pooling
    conv3a = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool2)
    act3a = Activation('relu')(conv3a)
    conv3b = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act3a)
    act3b = Activation('relu')(conv3b)
    conv3c = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act3b)
    act3c = Activation('relu')(conv3c)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(act3c)

    # Fourth block: Convolution, Convolution, Convolution, Max Pooling
    conv4a = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool3)
    act4a = Activation('relu')(conv4a)
    conv4b = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(act4a)
    act4b = Activation('relu')(conv4b)
    conv4c = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(act4b)
    act4c = Activation('relu')(conv4c)
    max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(act4c)

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(max_pool4)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model