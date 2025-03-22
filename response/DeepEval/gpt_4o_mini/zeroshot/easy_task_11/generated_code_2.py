import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = layers.Input(shape=(28, 28, 1))

    # Average Pooling layer with 5x5 window and 3x3 stride
    x = layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # 1x1 Convolutional layer
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # First fully connected layer
    x = layers.Dense(units=128, activation='relu')(x)

    # Dropout layer to reduce overfitting
    x = layers.Dropout(rate=0.5)(x)

    # Second fully connected layer
    x = layers.Dense(units=64, activation='relu')(x)

    # Output layer with softmax activation for multi-class classification
    output_layer = layers.Dense(units=10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model