import keras
import tensorflow as tf
from keras.layers import Input, SeparableConv2D, BatchNormalization, Lambda, Concatenate, AveragePooling2D, Conv2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions with different kernel sizes and batch normalization
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
        bn1 = BatchNormalization()(sep_conv1)
        
        sep_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
        bn2 = BatchNormalization()(sep_conv2)
        
        sep_conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_groups[2])
        bn3 = BatchNormalization()(sep_conv3)

        # Concatenate the outputs
        output_tensor = Concatenate()([bn1, bn2, bn3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 3x3 Average Pooling followed by 1x1 Convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 Convolution followed by two branches of 1x3 and 3x1 Convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_branch1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3)
        path3_branch2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3)
        path3_output = Concatenate()([path3_branch1, path3_branch2])

        # Path 4: 1x1 Convolution followed by 3x3 Convolution, then two branches of 1x3 and 3x1 Convolutions
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        path4_branch1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
        path4_branch2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
        path4_output = Concatenate()([path4_branch1, path4_branch2])

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3_output, path4_output])
        return output_tensor

    # Execute the blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Final classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model