import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the CIFAR-10 images (32x32 RGB images)
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # First average pooling layer with 1x1 pooling size and stride of 1
    x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)

    # Second average pooling layer with 2x2 pooling size and stride of 2
    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)

    # Third average pooling layer with 4x4 pooling size and stride of 4
    x3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten the outputs of the pooling layers
    x1_flat = layers.Flatten()(x1)
    x2_flat = layers.Flatten()(x2)
    x3_flat = layers.Flatten()(x3)

    # Concatenate the flattened outputs
    concatenated = layers.Concatenate()([x1_flat, x2_flat, x3_flat])

    # Fully connected layer 1
    x = layers.Dense(256, activation='relu')(concatenated)

    # Fully connected layer 2
    x = layers.Dense(128, activation='relu')(x)

    # Output layer with softmax activation for 10 classes (CIFAR-10)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()