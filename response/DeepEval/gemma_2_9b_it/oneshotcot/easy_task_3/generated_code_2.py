import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Convolution, Convolution, Max Pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_2)

    # Block 2: Convolution, Convolution, Max Pooling
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_2)

    # Block 3: Convolution, Convolution, Convolution, Max Pooling
    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3_3)

    # Block 4: Convolution, Convolution, Convolution, Max Pooling
    conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_2)
    max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4_3)

    flatten_layer = Flatten()(max_pool4)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model