import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main)

    conv_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_main)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main)

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)

    # Concatenate outputs
    concat = Concatenate()([max_pool_main, conv_branch])

    # Batch normalization
    batch_norm = BatchNormalization()(concat)

    # Flatten and fully connected layer
    flatten = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model