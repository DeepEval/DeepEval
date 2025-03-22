import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape

def dl_model():
    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define first block
    block_1 = []
    for scale in [1, 2, 4]:
        pooling = AveragePooling2D(pool_size=(scale, scale), strides=(scale, scale), padding='same')(input_layer)
        flatten = Flatten()(pooling)
        block_1.append(flatten)
    block_1 = Concatenate()(block_1)

    # Define second block
    block_2 = []
    for branch in range(3):
        conv = []
        for kernel_size in [1, 3, 7]:
            conv_layer = DepthwiseConv2D(kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_layer)
            conv.append(conv_layer)
        conv = Concatenate()(conv)
        pooling = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv)
        block_2.append(pooling)
    block_2 = Concatenate()(block_2)

    # Define output layers
    flatten_1 = Flatten()(block_1)
    flatten_2 = Flatten()(block_2)
    dense_1 = Dense(units=128, activation='relu')(flatten_1)
    dense_2 = Dense(units=128, activation='relu')(flatten_2)
    output_layer = Dense(units=10, activation='softmax')(Concatenate()([dense_1, dense_2]))

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model