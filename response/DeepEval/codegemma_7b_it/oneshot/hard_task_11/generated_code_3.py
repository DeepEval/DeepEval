import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    # Parallel branch
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
    conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
    conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
    maxpool1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(maxpool)

    # Concatenate outputs
    concat = Concatenate()([conv1, conv2, conv3, maxpool1x1, conv1x1, conv1x3, conv3x1])

    # 1x1 convolution
    conv_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Direct connection from input
    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion with main pathway
    output = Add()([conv_output, shortcut])

    # Classification layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model