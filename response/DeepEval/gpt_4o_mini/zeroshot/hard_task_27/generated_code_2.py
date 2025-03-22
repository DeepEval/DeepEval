import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = layers.Input(shape=(32, 32, 3))

    # Depthwise separable convolution layer with layer normalization
    x = layers.DepthwiseConv2D(kernel_size=(7, 7), padding='same')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # Fully connected layers with the same number of channels as input
    x = layers.GlobalAveragePooling2D()(x)  # Pooling to reduce the spatial dimensions
    x = layers.Dense(32, activation='relu')(x)  # First fully connected layer
    x = layers.Dense(32, activation='relu')(x)  # Second fully connected layer

    # Residual connection by adding the original input
    x = layers.Add()([x, layers.GlobalAveragePooling2D()(inputs)])

    # Output layer for 10 categories (CIFAR-10 classes)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to create the model
model = dl_model()
model.summary()