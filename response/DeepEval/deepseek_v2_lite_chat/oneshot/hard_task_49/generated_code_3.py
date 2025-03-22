import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():     

    # First block
    input_layer = Input(shape=(28, 28, 1))

    conv_pool_layers = [
        Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='sigmoid')(input_layer),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_pool_layers[-1]),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_pool_layers[-2])
    ]

    flattened_vectors = Flatten()(conv_pool_layers[-1])
    concatenated_vectors = Concatenate()(conv_pool_layers[:-1] + [flattened_vectors])

    dense1 = Dense(128, activation='relu')(concatenated_vectors)
    dense2 = Dense(64, activation='relu')(dense1)

    # Second block
    split_tensors = Lambda(lambda x: keras.layers.split(x, num_or_size_splits=4, axis=-1))(dense2)

    block1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_tensors[0])
    block2 = Conv2D(64, (3, 3), padding='same', activation='relu')(split_tensors[1])
    block3 = Conv2D(64, (5, 5), padding='same', activation='relu')(split_tensors[2])
    block4 = Conv2D(64, (7, 7), padding='same', activation='relu')(split_tensors[3])

    concatenated_outputs = Concatenate(axis=-1)([block1, block2, block3, block4])

    dense3 = Dense(10, activation='softmax')(concatenated_outputs)

    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model