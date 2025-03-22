import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    # CIFAR-10 images are 32x32 with 3 channels
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # First block: split channels and apply separable convolutions with different kernel sizes
    # Split input into three groups along the channel axis
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply separable convolutions with different kernel sizes (1x1, 3x3, and 5x5)
    conv1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layers[0])
    conv3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layers[1])
    conv5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layers[2])

    # Concatenate outputs from the separable convolutions
    concat1 = Concatenate()([conv1x1, conv3x3, conv5x5])

    # Second block: enhanced feature extraction using multiple branches
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(concat1)

    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(concat1)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concat1)

    # Concatenate outputs from all branches
    concat2 = Concatenate()([branch1, branch2, branch3])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(concat2)

    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(gap)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()