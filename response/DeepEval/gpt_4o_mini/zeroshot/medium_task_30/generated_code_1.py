import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First average pooling layer with 1x1 pooling window and stride of 1
    x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)

    # Second average pooling layer with 2x2 pooling window and stride of 2
    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)

    # Third average pooling layer with 4x4 pooling window and stride of 4
    x3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten the outputs of the pooling layers
    x1_flat = layers.Flatten()(x1)
    x2_flat = layers.Flatten()(x2)
    x3_flat = layers.Flatten()(x3)

    # Concatenate the flattened outputs
    concatenated = layers.Concatenate()([x1_flat, x2_flat, x3_flat])

    # Fully connected layers
    x = layers.Dense(512, activation='relu')(concatenated)
    x = layers.Dense(256, activation='relu')(x)

    # Output layer with softmax activation for classification (10 classes for CIFAR-10)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# model = dl_model()
# model.summary()