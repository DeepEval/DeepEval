import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer with 32 kernels
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    block_1_input = conv

    # Split input into two groups along the last dimension
    groups = Lambda(lambda x: tf.split(x, 2, axis=-1))(block_1_input)

    # Process first group
    group_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
    group_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group_1)
    group_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group_1)

    # Process second group
    group_2 = groups[1]

    # Concatenate outputs from both groups
    output = Concatenate()([group_1, group_2])

    # Block 2
    block_2_input = output

    # Reshape input into (height, width, groups, channels_per_group)
    shape = tf.shape(block_2_input)
    groups = tf.reshape(block_2_input, (shape[0], shape[1], 2, shape[2] // 2))

    # Swap third and fourth dimensions
    groups = tf.transpose(groups, [0, 1, 3, 2])

    # Reshape back to original shape
    output = tf.reshape(groups, (shape[0], shape[1], shape[2], shape[3] * 2))

    # Batch normalization
    output = BatchNormalization()(output)

    # Flatten and fully connected layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output)
    return model