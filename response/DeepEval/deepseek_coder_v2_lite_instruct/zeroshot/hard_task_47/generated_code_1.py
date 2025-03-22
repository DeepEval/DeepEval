import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Split the input into three groups with different kernel sizes
    split_1x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    split_3x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    split_5x5 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Depthwise separable convolutions with different kernel sizes
    def depthwise_separable_conv(x, kernel_size):
        depthwise_conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_initializer='he_uniform', depthwise_constraint=None, pointwise_initializer='he_uniform', pointwise_constraint=None)(x)
        bn = BatchNormalization()(depthwise_conv)
        act = Activation('relu')(bn)
        pointwise_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False, kernel_initializer='he_uniform')(act)
        bn_pointwise = BatchNormalization()(pointwise_conv)
        act_pointwise = Activation('relu')(bn_pointwise)
        return act_pointwise

    conv_1x1 = [depthwise_separable_conv(x, kernel_size=(1, 1)) for x in split_1x1]
    conv_3x3 = [depthwise_separable_conv(x, kernel_size=(3, 3)) for x in split_3x3]
    conv_5x5 = [depthwise_separable_conv(x, kernel_size=(5, 5)) for x in split_5x5]

    # Concatenate the outputs from the three groups
    concatenated = Concatenate(axis=-1)(conv_1x1 + conv_3x3 + conv_5x5)

    # Second block: Multiple branches for feature extraction
    def branch_block(x, filters):
        x = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return x

    branch1 = branch_block(concatenated, filters=64)
    branch2 = branch_block(concatenated, filters=64)
    branch3 = branch_block(concatenated, filters=64)
    branch4 = branch_block(concatenated, filters=64)
    branch5 = branch_block(concatenated, filters=64)

    # Concatenate the outputs from all branches
    concatenated_branches = Concatenate(axis=-1)([branch1, branch2, branch3, branch4, branch5])

    # Flatten and add fully connected layers
    x = Flatten()(concatenated_branches)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()