import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define input shape (CIFAR-10 images are 32x32 with 3 channels)
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Reshape the input tensor into groups
    groups = 3
    channels_per_group = input_shape[2] // groups
    reshaped = layers.Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(inputs)

    # Permute dimensions to shuffle channels
    permuted = layers.Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original input shape
    reshaped_back = layers.Reshape(input_shape)(permuted)

    # Add a convolutional layer for feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu')(reshaped_back)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Fully connected layer with softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()