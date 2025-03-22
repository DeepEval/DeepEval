import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv_1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv_1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1_1)
    max_pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1_2)

    # Block 2
    conv_2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling_1)
    conv_2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_2_1)
    conv_2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv_2_2)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2_3)

    # Flatten and Dense Layers
    flatten_layer = Flatten()(max_pooling_2)
    dense_1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model