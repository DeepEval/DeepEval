import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Convolutional layer followed by Batch Normalization and ReLU activation
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Initial feature maps
    initial_features = x

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Reshape the output to match the channels of the initial features
    x = layers.Reshape((1, 1, 32))(x)

    # Multiply with initial features to generate weighted feature maps
    x = layers.multiply([initial_features, x])

    # Concatenate with the input layer
    x = layers.Concatenate()([inputs, x])

    # Downsampling using 1x1 convolution and average pooling
    x = layers.Conv2D(32, (1, 1), padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Flatten and fully connected output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Load CIFAR-10 dataset (for testing purpose)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()