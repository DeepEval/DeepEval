import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the MNIST dataset (28x28 grayscale images)
    inputs = layers.Input(shape=(28, 28, 1))

    # Average pooling layer with a 5x5 window and a 3x3 stride
    x = layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(inputs)

    # 1x1 Convolutional layer
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # First fully connected layer
    x = layers.Dense(units=128, activation='relu')(x)

    # Dropout layer to mitigate overfitting
    x = layers.Dropout(rate=0.5)(x)

    # Second fully connected layer
    x = layers.Dense(units=64, activation='relu')(x)

    # Output layer for 10 classes (MNIST digits)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model with a suitable optimizer and loss function
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model