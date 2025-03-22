import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Dense, AveragePooling2D

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Split the input into three groups
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Depthwise separable convolutional layers with different kernel sizes
    def depthwise_separable_conv(x, kernel_size):
        x = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_multiplier=1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        return x

    conv1x1 = depthwise_separable_conv(split_1[0], kernel_size=(1, 1))
    conv3x3 = depthwise_separable_conv(split_1[1], kernel_size=(3, 3))
    conv5x5 = depthwise_separable_conv(split_1[2], kernel_size=(5, 5))

    # Concatenate the outputs from the three groups
    concat_1 = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Second block: Multiple branches for feature extraction
    def conv_branch(x, kernel_size):
        x = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        return x

    branch1 = conv_branch(concat_1, kernel_size=(1, 1))
    branch2 = conv_branch(concat_1, kernel_size=(3, 3))

    branch3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(concat_1)
    branch3 = BatchNormalization()(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = conv_branch(branch3, kernel_size=(3, 3))

    branch4 = AveragePooling2D(pool_size=(8, 8))(concat_1)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
    branch4 = BatchNormalization()(branch4)

    # Concatenate the outputs from all branches
    concat_2 = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    x = tf.keras.layers.Flatten()(concat_2)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()