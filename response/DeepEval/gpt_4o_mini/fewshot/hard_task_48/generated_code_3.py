import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, BatchNormalization, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Splitting the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Path 1: 1x1 separable convolution
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = BatchNormalization()(conv1)

        # Path 2: 3x3 separable convolution
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv2 = BatchNormalization()(conv2)

        # Path 3: 5x5 separable convolution
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        conv3 = BatchNormalization()(conv3)

        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Block 2
    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 convolution -> split into 1x3 and 3x1 convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        split_path3 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(path3)
        conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(split_path3[0])
        conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(split_path3[1])
        path3 = Concatenate()([conv1x3, conv3x1])

        # Path 4: 1x1 convolution -> 3x3 convolution -> split into 1x3 and 3x1 convolutions
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        split_path4 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(path4)
        conv1x3_2 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(split_path4[0])
        conv3x1_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(split_path4[1])
        path4 = Concatenate()([conv1x3_2, conv3x1_2])

        # Concatenate the outputs from all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    # Final classification layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model