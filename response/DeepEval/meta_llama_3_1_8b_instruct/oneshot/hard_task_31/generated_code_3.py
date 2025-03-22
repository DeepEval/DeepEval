import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Lambda, Dropout, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras import regularizers
import tensorflow as tf

def dl_model():     
    """
    A deep learning model for image classification using the CIFAR-10 dataset.

    This model consists of two main blocks. The first block features both a main path and a branch path.
    The main path begins with a <convolution, dropout> block to expand the width of the feature map,
    followed by a convolutional layer to restore the number of channels to same as those of input.
    In parallel, the branch path directly connects to the input. The outputs from both paths are then
    added to produce the output of this block. The second block split the input into three groups
    along the last dimension by encapsulating tf.split within Lambda layer, with each group using
    separable convolutional layers of varying kernel sizes (1x1, 3x3, and 5x5) to extract features.
    Each convolution is followed by a dropout layer to mitigate overfitting. The outputs from the
    three groups are concatenated to create a unified feature representation. After processing
    through these two blocks to extract features, the model outputs the final predictions via a
    flattening layer and a fully connected layer.
    """

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels
    # First block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    main_path_output = conv2
    # Branch path
    branch_path_output = input_layer
    # Add the outputs of the main and branch paths
    block_output = Add()([main_path_output, branch_path_output])

    # Second block
    def separable_conv(input_tensor, kernel_size):
        x = SeparableConv2D(filters=64, kernel_size=kernel_size, padding='same')(input_tensor)
        x = Dropout(0.2)(x)
        return x

    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block_output)
    # Extract features using separable convolutional layers
    conv1x1_output = separable_conv(split_layer[0], kernel_size=(1, 1))
    conv3x3_output = separable_conv(split_layer[1], kernel_size=(3, 3))
    conv5x5_output = separable_conv(split_layer[2], kernel_size=(5, 5))
    # Concatenate the outputs from the three groups
    concat_output = Concatenate()([conv1x1_output, conv3x3_output, conv5x5_output])
    # Batch normalization and global average pooling
    batch_norm = BatchNormalization()(concat_output)
    global_avg_pool = GlobalAveragePooling2D()(batch_norm)
    # Final layer
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model