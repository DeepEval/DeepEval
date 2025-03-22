import keras
from keras.layers import Input, Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First block: Split the input into three groups
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Depthwise separable convolution layers for feature extraction
    def depthwise_separable_conv_branch(input_tensor, kernel_size):
        return Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu', depth_multiplier=1)(input_tensor)

    branch1 = depthwise_separable_conv_branch(split_inputs[0], (1, 1))
    branch2 = depthwise_separable_conv_branch(split_inputs[1], (3, 3))
    branch3 = depthwise_separable_conv_branch(split_inputs[2], (5, 5))

    # Apply Batch Normalization to all branches
    branch1 = BatchNormalization()(branch1)
    branch2 = BatchNormalization()(branch2)
    branch3 = BatchNormalization()(branch3)

    # Concatenate outputs from the first block
    concatenated_block1 = Concatenate()([branch1, branch2, branch3])

    # Second block: Multiple branches for feature extraction
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_block1)
    branch5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch4)
    
    branch6 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_block1)
    branch6 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch6)
    branch6 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch6)
    branch6 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch6)

    branch7 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(concatenated_block1)

    # Concatenate outputs from the second block
    concatenated_block2 = Concatenate()([branch5, branch6, branch7])

    # Flatten and Fully Connected layers for classification
    flatten_layer = Flatten()(concatenated_block2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model