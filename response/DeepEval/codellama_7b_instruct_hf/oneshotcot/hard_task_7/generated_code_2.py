import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Initial Convolutional Layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Block 1: Splitting the input into two groups
    split_layer = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv)
    group1 = split_layer[0]
    group2 = split_layer[1]

    # Block 1: First group operations
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)

    # Block 1: Second group operations
    group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)

    # Block 1: Merging the outputs from both groups
    merged_output = Concatenate()([group1, group2])

    # Block 2: Convolutional Layer
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(merged_output)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Block 2: Channel Shuffling
    channel_shuffling_layer = Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]))(max_pooling)

    # Block 2: Flattening the output
    flatten_layer = Flatten()(channel_shuffling_layer)

    # Block 2: Fully Connected Layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model