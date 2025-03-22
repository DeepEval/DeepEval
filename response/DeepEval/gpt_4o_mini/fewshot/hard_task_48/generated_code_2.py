import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    def block_1(input_tensor):
        # Split input into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Group 1: 1x1 Separable Convolution
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        bn1 = BatchNormalization()(conv1)
        
        # Group 2: 3x3 Separable Convolution
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        bn2 = BatchNormalization()(conv2)
        
        # Group 3: 5x5 Separable Convolution
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        bn3 = BatchNormalization()(conv3)

        # Concatenate the outputs of the three groups
        output_tensor = Concatenate()([bn1, bn2, bn3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 3x3 Average Pooling followed by 1x1 Convolution
        path2_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2_pool)

        # Path 3: 1x1 Convolution followed by split into 1x3 and 3x1 convolutions
        path3_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_split = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(path3_conv)
        path3_conv1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3_split[0])
        path3_conv2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3_split[1])
        path3 = Concatenate()([path3_conv1, path3_conv2])

        # Path 4: 1x1 Convolution followed by 3x3 Convolution, then split into 1x3 and 3x1 convolutions
        path4_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path4_conv1)
        path4_split = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(path4_conv2)
        path4_conv3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4_split[0])
        path4_conv4 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4_split[1])
        path4 = Concatenate()([path4_conv3, path4_conv4])

        # Concatenate outputs of the four paths
        output_tensor = Concatenate()([path1, path2_conv, path3, path4])
        return output_tensor

    # Construct the model
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model