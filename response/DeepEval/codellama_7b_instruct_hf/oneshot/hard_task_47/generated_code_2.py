import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: feature extraction
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    # Apply depthwise separable convolutional layers with different kernel sizes (1x1, 3x3, 5x5)
    conv1_layer = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv2_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv3_layer = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])
    # Batch normalization
    bn1_layer = BatchNormalization()(conv1_layer)
    bn2_layer = BatchNormalization()(conv2_layer)
    bn3_layer = BatchNormalization()(conv3_layer)
    # Concatenate the outputs from the three branches
    concat_layer = Concatenate()([bn1_layer, bn2_layer, bn3_layer])

    # Second block: feature extraction
    # Apply multiple branches for feature extraction
    branch1_layer = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat_layer)
    branch2_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat_layer)
    branch3_layer = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same')(concat_layer)
    branch4_layer = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')(concat_layer)
    branch5_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat_layer)
    # Average pooling
    pooling_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(concat_layer)
    # Concatenate the outputs from all branches
    concat_layer = Concatenate()([branch1_layer, branch2_layer, branch3_layer, branch4_layer, branch5_layer, pooling_layer])

    # Flatten the output
    flatten_layer = Flatten()(concat_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model