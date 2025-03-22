import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # First block: Depthwise separable convolutions
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutions with different kernel sizes
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])

    # Batch normalization after each convolution
    depthwise_conv1 = BatchNormalization()(depthwise_conv1)
    depthwise_conv2 = BatchNormalization()(depthwise_conv2)
    depthwise_conv3 = BatchNormalization()(depthwise_conv3)

    # Concatenate the outputs
    concatenated = Concatenate()([depthwise_conv1, depthwise_conv2, depthwise_conv3])

    # Second block: Multiple branches for feature extraction
    branch1 = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(concatenated)
    ])

    branch2 = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated),
        Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(concatenated),
        Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(concatenated),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(concatenated)
    ])

    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(concatenated)

    # Concatenate all branches
    final_output = Concatenate()([branch1, branch2, branch3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(final_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model