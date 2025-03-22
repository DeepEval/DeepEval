import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape as per CIFAR-10 dataset
    input_shape = (32, 32, 3)
    num_classes = 10  # CIFAR-10 has 10 classes

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the last dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layers = Lambda(split_input)(inputs)

    # Apply depthwise separable convolutions with different kernel sizes
    conv1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layers[0])
    conv3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layers[1])
    conv5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layers[2])

    # Concatenate the outputs from the separable convolutions
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])

    # Flatten the fused features
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(flattened)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model