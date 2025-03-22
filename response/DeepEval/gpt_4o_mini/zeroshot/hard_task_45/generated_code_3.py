import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB images
    inputs = layers.Input(shape=input_shape)

    # First Block
    # Split the input into three groups
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Apply depthwise separable convolutions with different kernel sizes
    conv1 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(splits[0])
    conv2 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(splits[1])
    conv3 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(splits[2])

    # Concatenate outputs from the three groups
    block1_output = layers.Concatenate()([conv1, conv2, conv3])

    # Second Block
    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(block1_output)

    # Branch 2: <1x1 Convolution>
    branch2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(block1_output)

    # Branch 3: 3x3 Convolution
    branch3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(block1_output)

    # Branch 4: 3x3 Convolution
    branch4 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(block1_output)

    # Branch 5: <1x1 Convolution, 3x3 Convolution>
    branch5 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(block1_output)
    branch5 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch5)

    # Branch 6: <Max Pooling, 1x1 Convolution>
    branch6 = layers.MaxPooling2D(pool_size=(2, 2))(block1_output)
    branch6 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(branch6)

    # Concatenate outputs from all branches
    block2_output = layers.Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6])

    # Flattening and Fully Connected Layer
    flatten = layers.Flatten()(block2_output)
    outputs = layers.Dense(10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model