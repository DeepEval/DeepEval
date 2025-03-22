import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_input)
    block1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_conv1)
    block1_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_conv2)
    block1_output = Concatenate()([block1_conv1, block1_conv2, block1_conv3])

    # Block 2
    block2_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_input)
    block2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_conv1)
    block2_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_conv2)
    block2_output = Concatenate()([block2_conv1, block2_conv2, block2_conv3])

    # Parallel Branch
    parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs
    output = Concatenate()([block1_output, block2_output, parallel_branch])

    # Flatten and fully connected layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model