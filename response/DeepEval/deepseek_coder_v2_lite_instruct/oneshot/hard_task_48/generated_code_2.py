import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        # Split the input into three groups
        split_1 = Lambda(lambda z: z[:, :11, :11, :])(x)
        split_2 = Lambda(lambda z: z[:, 7:, 7:, :])(x)
        split_3 = Lambda(lambda z: z[:, :, :, :])(x)

        # Extract features through separable convolutional layers
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_1)
        conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_2)
        conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_3)

        # Batch normalization
        conv_1x1 = BatchNormalization()(conv_1x1)
        conv_3x3 = BatchNormalization()(conv_3x3)
        conv_5x5 = BatchNormalization()(conv_5x5)

        # Concatenate the outputs of the three groups
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated

    block1_output = block1(input_layer)

    # Block 2
    def block2(x):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)

        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 convolution followed by two sub-paths
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
        sub_path3_1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3)
        sub_path3_2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([sub_path3_1, sub_path3_2])

        # Path 4: 1x1 convolution followed by 3x3 convolution and two sub-paths
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        sub_path4_1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4)
        sub_path4_2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([sub_path4_1, sub_path4_2])

        # Concatenate the outputs of the four paths
        concatenated = Concatenate()([path1, path2, path3, path4])
        return concatenated

    block2_output = block2(block1_output)

    # Flatten the result and add a fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model