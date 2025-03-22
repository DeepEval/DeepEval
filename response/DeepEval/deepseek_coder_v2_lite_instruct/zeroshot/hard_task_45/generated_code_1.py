import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Depthwise separable convolutions with varying kernel sizes
    def depthwise_separable_conv(x, kernel_size):
        # Depthwise convolution
        depthwise_conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_initializer='glorot_uniform', depthwise_constraint=None, dilation_rate=(1, 1))(x)
        # Pointwise convolution
        pointwise_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False, activation='relu')(depthwise_conv)
        return pointwise_conv

    # Split the input into three groups and apply depthwise separable convolutions
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    conv_outputs = [depthwise_separable_conv(split_x, kernel_size) for split_x in split_1]
    concatenated_features = Concatenate(axis=-1)(conv_outputs)

    # Second block: Multiple branches for feature extraction
    def feature_extraction_branch(x, config):
        features = []
        for layer in config:
            if layer == '1x1':
                x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
            elif layer == '<1x1':
                x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
            elif layer == '3x3':
                x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
            elif layer == 'maxpool':
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x

    # Define branches with different configurations
    branches = [
        feature_extraction_branch(concatenated_features, ['1x1', '3x3']),
        feature_extraction_branch(concatenated_features, ['<1x1', '3x3']),
        feature_extraction_branch(concatenated_features, ['maxpool', '1x1']),
        feature_extraction_branch(concatenated_features, ['1x1']),
        feature_extraction_branch(concatenated_features, ['<1x1']),
        feature_extraction_branch(concatenated_features, ['3x3'])
    ]

    # Concatenate outputs from all branches
    fused_features = Concatenate(axis=-1)(branches)

    # Flatten the output and pass it through a fully connected layer
    x = Flatten()(fused_features)
    outputs = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()