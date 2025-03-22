import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 color channel
    inputs = layers.Input(shape=input_shape)

    # Main path
    main_path = inputs
    for _ in range(3):  # Repeat the block 3 times
        # Apply separable convolution followed by ReLU activation
        x = layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same')(main_path)
        x = layers.ReLU()(x)
        
        # Concatenate the original input with the convolved feature map
        main_path = layers.Concatenate()([main_path, x])
    
    # Branch path
    branch_path = layers.Conv2D(32, kernel_size=(1, 1), padding='same')(main_path)

    # Fuse the two paths with an addition operation
    fused = layers.Add()([main_path, branch_path])

    # Flatten the features
    flatten = layers.Flatten()(fused)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flatten)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()