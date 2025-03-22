from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, SeparableConv2D, BatchNormalization, concatenate, Flatten, Dense
from keras.applications.cifar10 import Cifar10Data

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_outputs = []
    for i in range(3):
        # Split input into three groups
        split_layer = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)
        # Extract features through separable convolutional layers with different kernel sizes
        conv1_layer = SeparableConv2D(32, (1, 1), activation='relu')(split_layer)
        conv3_layer = SeparableConv2D(32, (3, 3), activation='relu')(split_layer)
        conv5_layer = SeparableConv2D(32, (5, 5), activation='relu')(split_layer)
        # Batch normalization to enhance model performance
        norm1_layer = BatchNormalization()(conv1_layer)
        norm3_layer = BatchNormalization()(conv3_layer)
        norm5_layer = BatchNormalization()(conv5_layer)
        # Concatenate outputs of three groups
        block1_outputs.append(concatenate([norm1_layer, norm3_layer, norm5_layer]))

    # Block 2
    block2_outputs = []
    for i in range(4):
        # Process input through 1x1 convolution
        conv1_layer = Conv2D(64, (1, 1), activation='relu')(input_layer)
        # Perform 3x3 average pooling followed by 1x1 convolution
        pool3_layer = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(conv1_layer)
        conv3_layer = Conv2D(64, (1, 1), activation='relu')(pool3_layer)
        # Split into two sub-paths that use 1x3 and 3x1 convolutions respectively
        path1_layer = Conv2D(64, (1, 3), activation='relu')(conv3_layer)
        path2_layer = Conv2D(64, (3, 1), activation='relu')(conv3_layer)
        # Concatenate results
        block2_outputs.append(concatenate([path1_layer, path2_layer]))

    # Concatenate outputs of block 1 and block 2
    output_layer = concatenate(block1_outputs + block2_outputs)

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(output_layer)
    dense_layer = Dense(10, activation='softmax')(flatten_layer)

    # Create model
    model = Model(input_layer, dense_layer)

    return model