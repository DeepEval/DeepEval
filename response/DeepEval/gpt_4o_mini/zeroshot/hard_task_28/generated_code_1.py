import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Main path
    # Depthwise convolution
    main_path = layers.DepthwiseConv2D(kernel_size=(7, 7), padding='same')(inputs)
    
    # Layer normalization
    main_path = layers.LayerNormalization()(main_path)

    # Pointwise convolutions
    main_path = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)
    main_path = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path (directly connects to the input)
    branch_path = inputs

    # Combine both paths
    combined = layers.Add()([main_path, branch_path])

    # Flatten the combined output
    flattened = layers.Flatten()(combined)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()  # Display the model summary