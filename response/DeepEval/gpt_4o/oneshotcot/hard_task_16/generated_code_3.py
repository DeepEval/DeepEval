import keras
from keras.layers import Input, Conv2D, Dense, Lambda, Concatenate, GlobalMaxPooling2D, Multiply, Add, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input into 3 parts along the last dimension
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # First group: 1x1 Convolution
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1_2)

        # Second group: 1x1 Convolution
        conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split2)
        conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2_2)

        # Third group: 1x1 Convolution
        conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split3)
        conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv3_2)

        # Concatenate outputs from three groups
        output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])

        return output_tensor

    block1_output = block1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block2(adjusted_tensor, original_input):
        # Global max pooling
        global_pool = GlobalMaxPooling2D()(adjusted_tensor)

        # Fully connected layers to generate channel weights
        dense1 = Dense(units=64, activation='relu')(global_pool)
        dense2 = Dense(units=3, activation='sigmoid')(dense1)

        # Reshape channel weights to match adjusted output dimensions
        channel_weights = Reshape((1, 1, 3))(dense2)

        # Multiply adjusted output by channel weights
        main_path_output = Multiply()([adjusted_tensor, channel_weights])

        # Direct branch from the original input
        branch_output = original_input

        # Add main path and branch outputs
        added_output = Add()([main_path_output, branch_output])

        return added_output

    block2_output = block2(transition_conv, input_layer)

    # Fully connected layer for classification
    flatten_layer = GlobalMaxPooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model