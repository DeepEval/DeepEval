import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # Main path
    # Convolutional layers (you can adjust the number of filters and kernel sizes as needed)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Global average pooling
    pooled_output = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers for generating weights
    dense_1 = layers.Dense(32, activation='relu')(pooled_output)
    dense_2 = layers.Dense(32 * 32 * 3, activation='sigmoid')(dense_1)  # Output size matches the input channels
    reshaped_weights = layers.Reshape((32, 32, 3))(dense_2)
    
    # Element-wise multiplication with the original feature map
    weighted_output = layers.multiply([x, reshaped_weights])

    # Branch path
    # Convolution to adjust the channel size
    branch_output = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Adding the outputs of the main and branch paths
    combined_output = layers.add([weighted_output, branch_output])

    # Passing the combined result through fully connected layers for classification
    combined_flattened = layers.Flatten()(combined_output)
    fc_1 = layers.Dense(128, activation='relu')(combined_flattened)
    fc_2 = layers.Dense(64, activation='relu')(fc_1)
    outputs = layers.Dense(10, activation='softmax')(fc_2)  # 10 classes for CIFAR-10

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()