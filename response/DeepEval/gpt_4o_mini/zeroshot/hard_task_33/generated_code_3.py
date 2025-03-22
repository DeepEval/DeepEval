import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel (grayscale)
    inputs = layers.Input(shape=input_shape)

    def create_branch(inputs):
        x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)  # 1x1 Convolution
        x = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)  # Depthwise separable convolution
        x = layers.Conv2D(32, (1, 1), activation='relu')(x)  # 1x1 Convolution
        x = layers.add([x, inputs])  # Residual connection
        return x

    # Create three branches
    branch1 = create_branch(inputs)
    branch2 = create_branch(inputs)
    branch3 = create_branch(inputs)

    # Concatenate the outputs from the three branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()