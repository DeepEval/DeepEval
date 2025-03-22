import keras
import tensorflow as tf
from keras.layers import Input, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Lambda, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Separable convolutions with different kernel sizes for each group
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])

        # Apply batch normalization
        bn1 = BatchNormalization()(conv1)
        bn2 = BatchNormalization()(conv2)
        bn3 = BatchNormalization()(conv3)

        # Concatenate the outputs
        output_tensor = Concatenate()([bn1, bn2, bn3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2_pool)

        # Path 3: 1x1 convolution, split into two sub-paths with 1x3 and 3x1 convolutions
        path3_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_1 = SeparableConv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3_1x1)
        path3_2 = SeparableConv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3_1x1)
        path3 = Concatenate()([path3_1, path3_2])

        # Path 4: 1x1 convolution, followed by 3x3 convolution, split into two sub-paths with 1x3 and 3x1 convolutions
        path4_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path4_1x1)
        path4_1 = SeparableConv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4_3x3)
        path4_2 = SeparableConv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4_3x3)
        path4 = Concatenate()([path4_1, path4_2])

        # Concatenate the outputs of the four paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Create the model architecture using the defined blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Final classification layers
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model