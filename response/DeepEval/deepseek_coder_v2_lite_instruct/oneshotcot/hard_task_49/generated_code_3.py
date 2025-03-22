import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model

def dl_model():
    # First Block
    input_layer = Input(shape=(28, 28, 1))

    # Pooling Layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)

    # Flatten and Concatenate
    concat_layer = Concatenate()([Flatten()(pool1), Flatten()(pool2), Flatten()(pool3)])

    # Fully Connected Layer and Reshape
    fc_layer = Dense(units=128, activation='relu')(concat_layer)
    reshape_layer = Reshape((4, 4, 8))(fc_layer)  # Assuming output shape is (4, 4, 8)

    # Second Block
    def second_block(input_tensor):
        # Split the input into four groups
        split_layers = []
        for i in range(4):
            split_layer = Lambda(lambda x, idx: x[:, :, :, idx * (28 // (2 ** (i + 1))): (idx + 1) * (28 // (2 ** (i + 1)))],
                                 output_shape=(14 // (2 ** i), 14 // (2 ** i), 28 // (2 ** (i + 1))),
                                 arguments={'idx': i})(input_tensor)
            split_layers.append(split_layer)

        # Depthwise separable convolutions
        conv_layers = []
        for i, split_layer in enumerate(split_layers):
            if i == 0:
                conv_layer = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer)
            elif i == 1:
                conv_layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer)
            elif i == 2:
                conv_layer = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer)
            elif i == 3:
                conv_layer = Conv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split_layer)
            conv_layers.append(conv_layer)

        # Concatenate the outputs
        concat_layer = Concatenate()(conv_layers)

        # Flatten the result
        flatten_layer = Flatten()(concat_layer)

        return flatten_layer

    second_block_output = second_block(reshape_layer)

    # Fully Connected Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(second_block_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model