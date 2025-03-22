import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    # Each group will be processed by a depthwise separable convolutional layer
    # with a different kernel size
    split_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Construct the first block
    block1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_input[0])
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(block1)
    block1 = BatchNormalization()(block1)

    # Construct the second block
    block2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_input[1])
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseSeparableConv2D(filters=128, kernel_size=(1, 7), strides=(1, 1), padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseSeparableConv2D(filters=256, kernel_size=(7, 1), strides=(1, 1), padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseSeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2)
    block2 = BatchNormalization()(block2)

    # Construct the third block
    block3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_input[2])
    block3 = BatchNormalization()(block3)
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block3)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseSeparableConv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(block3)
    block3 = BatchNormalization()(block3)

    # Concatenate the outputs from the three blocks
    concatenated_blocks = Concatenate()([block1, block2, block3])

    # Flatten the concatenated blocks
    flattened = Flatten()(concatenated_blocks)

    # Add two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model