import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    max_pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pooling_2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(max_pooling_1)
    max_pooling_3 = MaxPooling2D(pool_size=(8, 8), strides=8, padding='valid')(max_pooling_2)
    flatten_1 = Flatten()(max_pooling_3)

    # Block 2
    conv_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_3)
    flatten_2 = Flatten()(max_pooling_4)

    # Fully connected layer and reshape operation
    dense_1 = Dense(units=128, activation='relu')(flatten_1)
    reshape_1 = Flatten()(dense_1)

    # Concatenation of block 1 and block 2
    concatenated = Concatenate()([reshape_1, flatten_2])

    # Dense layers for classification
    dense_2 = Dense(units=10, activation='softmax')(concatenated)

    model = keras.Model(inputs=input_layer, outputs=dense_2)

    return model