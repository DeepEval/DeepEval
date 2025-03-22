import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv1_main = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    conv3_main = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2_main)
    maxpool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3_main)

    # Parallel pathway
    conv1_parallel = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2_parallel = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1_parallel)
    conv3_parallel = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2_parallel)
    maxpool_parallel = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3_parallel)

    # Concatenate outputs and 1x1 convolution
    concat = Concatenate()([maxpool_main, maxpool_parallel])
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Direct connection and additive fusion
    shortcut = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output = Add()([conv4, shortcut])

    # Classification layers
    batch_norm = BatchNormalization()(output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model