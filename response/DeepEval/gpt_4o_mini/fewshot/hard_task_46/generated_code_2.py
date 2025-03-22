import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Block 1: Splitting input into three groups and applying separable convolutions
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Separable convolutions for each group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])

        # Concatenating outputs from the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2: Multiple branches for enhanced feature extraction
    def block_2(input_tensor):
        # Branch 1: Simple 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        # Branch 3: Max pooling
        branch3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)

        # Concatenating outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Processing input through both blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Global average pooling and dense layer for classification
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model