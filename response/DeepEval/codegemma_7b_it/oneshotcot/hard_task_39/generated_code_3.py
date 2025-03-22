import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    pool1_flatten = Flatten()(pool1)
    pool2_flatten = Flatten()(pool2)
    pool3_flatten = Flatten()(pool3)
    block1_output = Concatenate()([pool1_flatten, pool2_flatten, pool3_flatten])

    # Block 2
    input_block2 = Reshape((1, 1, block1_output.shape[1]))(block1_output)
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_block2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_block2)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_block2)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_block2)
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Output layer
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model