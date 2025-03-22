import keras
from keras.layers import Input, Conv2D, SeparableConv2D, AveragePooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting the input and applying separable convolutions
    def block1(input_tensor):
        # Split the input into three groups along the channel axis
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_layer[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split_layer[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split_layer[2])

        path1 = BatchNormalization()(path1)
        path2 = BatchNormalization()(path2)
        path3 = BatchNormalization()(path3)

        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block1_output = block1(input_layer)

    # Block 2: Four parallel paths
    def block2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)

        # Path 2: 3x3 Average Pooling followed by 1x1 Convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(path2)

        # Path 3: 1x1 Convolution followed by 1x3 and 3x1 Convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3_1 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='same')(path3)
        path3_2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(path3)
        path3_output = Concatenate()([path3_1, path3_2])

        # Path 4: 1x1 Convolution followed by 3x3 Convolution and then split into 1x3 and 3x1
        path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path4)
        path4_1 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='same')(path4)
        path4_2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(path4)
        path4_output = Concatenate()([path4_1, path4_2])

        output_tensor = Concatenate()([path1, path2, path3_output, path4_output])

        return output_tensor

    block2_output = block2(block1_output)

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model