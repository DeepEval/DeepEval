import keras
from keras.layers import Input, Conv2D, AveragePooling2D, SeparableConv2D, BatchNormalization, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Split the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Process each group through a separable convolutional layer with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu')(groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu')(groups[2])

        # Apply batch normalization
        conv1 = BatchNormalization()(conv1)
        conv2 = BatchNormalization()(conv2)
        conv3 = BatchNormalization()(conv3)

        # Concatenate the outputs of the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)

        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(input_tensor))

        # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path3)
        path3_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path3)
        path3 = Concatenate()([path3_1, path3_2])

        # Path 4: 1x1 convolution followed by 3x3 convolution, then 1x3 and 3x1 convolutions
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path4)
        path4_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path4)
        path4_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path4)
        path4 = Concatenate()([path4_1, path4_2])

        # Concatenate the outputs of the four paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Apply Block 1 to the input layer
    block1_output = block_1(input_tensor=input_layer)

    # Apply Block 2 to the output of Block 1
    block2_output = block_2(input_tensor=block1_output)

    # Flatten the output of Block 2 and pass it through a fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model