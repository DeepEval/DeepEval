import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global average pooling
    x_global_pool = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x_fc1 = layers.Dense(32, activation='relu')(x_global_pool)
    x_fc2 = layers.Dense(32, activation='relu')(x_fc1)

    # Reshape the output to match the initial feature map shape
    x_reshaped = layers.Reshape((1, 1, 32))(x_fc2)

    # Multiply with the initial features to generate weighted feature maps
    x_weighted = layers.multiply([x, x_reshaped])

    # Concatenate with the input layer
    x_concat = layers.concatenate([input_layer, x_weighted])

    # Downsampling using 1x1 convolution
    x_conv = layers.Conv2D(16, (1, 1), padding='same')(x_concat)
    
    # Average pooling to reduce dimensionality
    x_pool = layers.AveragePooling2D(pool_size=(2, 2))(x_conv)

    # Flattening the features for the final fully connected layer
    x_flatten = layers.Flatten()(x_pool)

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(x_flatten)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()  # Display the model summary