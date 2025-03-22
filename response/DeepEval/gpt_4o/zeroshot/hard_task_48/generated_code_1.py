from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Block 1
    # Split input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Path for 1x1 separable convolution
    conv1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layer[0])
    bn1x1 = BatchNormalization()(conv1x1)

    # Path for 3x3 separable convolution
    conv3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    bn3x3 = BatchNormalization()(conv3x3)

    # Path for 5x5 separable convolution
    conv5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])
    bn5x5 = BatchNormalization()(conv5x5)

    # Concatenate the outputs of the three paths
    block1_output = Concatenate()([bn1x1, bn3x3, bn5x5])

    # Block 2
    # Path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), padding='same', activation='relu')(block1_output)

    # Path 2: 3x3 average pooling followed by 1x1 convolution
    path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2 = Conv2D(32, (1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
    path3 = Conv2D(32, (1, 1), padding='same', activation='relu')(block1_output)
    path3_1x3 = Conv2D(32, (1, 3), padding='same', activation='relu')(path3)
    path3_3x1 = Conv2D(32, (3, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution, 3x3 convolution, followed by 1x3 and 3x1 convolutions
    path4 = Conv2D(32, (1, 1), padding='same', activation='relu')(block1_output)
    path4 = Conv2D(32, (3, 3), padding='same', activation='relu')(path4)
    path4_1x3 = Conv2D(32, (1, 3), padding='same', activation='relu')(path4)
    path4_3x1 = Conv2D(32, (3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1x3, path4_3x1])

    # Concatenate the outputs of the four paths
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten and fully connected layer for classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model