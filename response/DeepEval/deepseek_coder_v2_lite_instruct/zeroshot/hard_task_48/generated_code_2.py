import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Add, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # Block 1
    # Split the input into three groups
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)

    # Extract features through separable convolutional layers with different kernel sizes
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_1[0])
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_1[1])
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_1[2])

    # Batch normalization
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_5x5 = BatchNormalization()(conv_5x5)

    # Concatenate the outputs of the three groups
    block1_output = Concatenate(axis=3)([conv_1x1, conv_3x3, conv_5x5])

    # Block 2
    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Path 2: 3x3 average pooling followed by 1x1 convolution
    path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output))

    # Path 3: 1x1 convolution followed by two sub-paths
    path3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3_1)
    path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3_1)
    path3 = Concatenate(axis=3)([path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution followed by two sub-paths
    path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
    path4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate(axis=3)([path4_1x3, path4_3x1])

    # Concatenate the outputs of the four paths
    block2_output = Concatenate(axis=3)([path1, path2, path3, path4])

    # Flatten and fully connected layer
    flatten = Flatten()(block2_output)
    output = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    return model