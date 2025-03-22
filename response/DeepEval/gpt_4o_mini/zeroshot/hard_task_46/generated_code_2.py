import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = Input(shape=input_shape)

    # First block
    # Split input into 3 channels
    split_inputs = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Apply separable convolutions with different kernel sizes
    conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_inputs[0])
    conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])
    conv3 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate outputs from the three separable convolutions
    block1_output = Concatenate()([conv1, conv2, conv3])

    # Second block
    # 1. 3x3 convolution
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(block1_output)

    # 2. Series of layers: 1x1 conv -> 3x3 conv -> 3x3 conv
    branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # 3. Max pooling branch
    branch3 = MaxPooling2D(pool_size=(2, 2))(block1_output)

    # Concatenate outputs from all branches
    block2_output = Concatenate()([branch1, branch2, branch3])

    # Global Average Pooling
    gap_output = GlobalAveragePooling2D()(block2_output)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(gap_output)  # CIFAR-10 has 10 classes

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model