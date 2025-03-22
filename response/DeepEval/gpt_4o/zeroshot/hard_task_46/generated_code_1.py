import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block: split, separable convolutions, and concatenate
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_channels)(input_layer)

    conv1 = SeparableConv2D(32, (1, 1), activation='relu', padding='same')(split_layer[0])
    conv2 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split_layer[1])
    conv3 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split_layer[2])

    concat_block1 = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Second block: multiple branches and concatenate
    # Branch 1: a 3x3 convolution
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_block1)

    # Branch 2: 1x1 convolution followed by two 3x3 convolutions
    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat_block1)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: max pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concat_block1)

    # Concatenate all branches
    concat_block2 = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Global average pooling
    gap = GlobalAveragePooling2D()(concat_block2)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(gap)  # CIFAR-10 has 10 classes

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model