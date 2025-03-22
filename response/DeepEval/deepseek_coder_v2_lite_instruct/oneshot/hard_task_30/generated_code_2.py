import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(x):
        # Main path
        main_path = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        main_path = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)
        # Branch path
        branch_path = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        # Add paths
        output_tensor = tf.add(main_path, branch_path)
        return output_tensor

    block1_output = first_block(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = MaxPooling2D((2, 2))(block1_output)

    # Second block
    def second_block(x):
        # Split input into three groups
        split_1 = Lambda(lambda z: z[:, :16, :16, :])(x)
        split_2 = Lambda(lambda z: z[:, 16:, :16, :])(x)
        split_3 = Lambda(lambda z: z[:, :16, 16:, :])(x)

        # Extract features using depthwise separable convolutions
        depthwise_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_1)
        depthwise_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu', groups=3)(split_2)
        depthwise_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', groups=3)(split_3)

        # Concatenate outputs
        concatenated_output = Concatenate()([depthwise_1x1, depthwise_3x3, depthwise_5x5])
        return concatenated_output

    block2_output = second_block(block1_output)
    block2_output = BatchNormalization()(block2_output)
    block2_output = MaxPooling2D((2, 2))(block2_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model