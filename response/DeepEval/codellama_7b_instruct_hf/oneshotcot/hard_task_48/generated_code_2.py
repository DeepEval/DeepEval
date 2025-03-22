import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split input into three groups using Lambda layer
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)

    # Path 1: Separable convolution with kernel size 1x1
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split_layer[0])
    batch_norm1 = BatchNormalization()(conv1)

    # Path 2: Separable convolution with kernel size 3x3
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(split_layer[1])
    batch_norm2 = BatchNormalization()(conv2)

    # Path 3: Separable convolution with kernel size 5x5
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(split_layer[2])
    batch_norm3 = BatchNormalization()(conv3)

    # Concatenate outputs from all three paths
    concat_layer = Concatenate()([batch_norm1, batch_norm2, batch_norm3])

    # Block 2: Four parallel branches
    parallel_branches = []
    for i in range(4):
        branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(concat_layer)
        branch = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(branch)
        parallel_branches.append(branch)

    # Concatenate outputs from all four parallel branches
    parallel_concatenate_layer = Concatenate()(parallel_branches)

    # Flatten and add fully connected layers
    flatten_layer = Flatten()(parallel_concatenate_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model