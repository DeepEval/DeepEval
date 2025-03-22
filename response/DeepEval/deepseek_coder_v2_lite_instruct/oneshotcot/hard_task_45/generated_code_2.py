import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(x):
        # Split the input into three groups
        split_1 = Lambda(lambda tensor: tensor[:, :11, :, :])(x)
        split_2 = Lambda(lambda tensor: tensor[:, 11:22, :, :])(x)
        split_3 = Lambda(lambda tensor: tensor[:, 22:, :, :])(x)

        # Apply depthwise separable convolutions
        conv_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_1)
        conv_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(split_2)
        conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(split_3)

        # Concatenate the outputs
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated

    first_block_output = first_block(input_layer)
    batch_norm_1 = BatchNormalization()(first_block_output)

    # Second block
    def second_block(x):
        # Branch 1: 1x1 convolution
        branch_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)

        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch_2_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        branch_2_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2_1)

        # Branch 3: 3x3 convolution
        branch_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

        # Branch 4: <max pooling, 1x1 convolution>
        branch_4_1 = MaxPooling2D((3, 3), strides=1, padding='same')(x)
        branch_4_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(branch_4_1)

        # Branch 5: <1x1 convolution, 3x3 convolution>
        branch_5_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        branch_5_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_5_1)

        # Branch 6: <1x1 convolution, 3x3 convolution>
        branch_6_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        branch_6_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_6_1)

        # Concatenate all branches
        concatenated = Concatenate()([branch_1, branch_2_2, branch_3, branch_4_2, branch_5_2, branch_6_2])
        return concatenated

    second_block_output = second_block(batch_norm_1)
    batch_norm_2 = BatchNormalization()(second_block_output)
    flatten_layer = Flatten()(batch_norm_2)
    dense_layer = Dense(units=100, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model