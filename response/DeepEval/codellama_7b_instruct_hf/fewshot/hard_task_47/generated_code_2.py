import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense, Concatenate, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Split the input into three groups along the last dimension
    input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutional layers with different kernel sizes
    # (1x1, 3x3, 5x5) for feature extraction
    conv1 = DepthwiseSeparableConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    conv2 = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    conv3 = DepthwiseSeparableConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])

    # Batch normalization
    batch_norm = BatchNormalization()(conv1)

    # Concatenate the outputs from all branches
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Second block
    # Apply multiple branches for feature extraction
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch5 = AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4, branch5])

    # Flatten the output
    flattened = Flatten()(concatenated)

    # Apply two fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model