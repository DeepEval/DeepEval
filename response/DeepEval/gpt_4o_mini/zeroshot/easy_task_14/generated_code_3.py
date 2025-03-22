import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Global Average Pooling layer to compress the input features
    x = layers.GlobalAveragePooling2D()(inputs)

    # Fully connected layer to learn the channel correlations
    x = layers.Dense(input_shape[-1], activation='relu')(x)
    x = layers.Dense(input_shape[-1], activation='relu')(x)

    # Reshape the weights to align with the input shape
    channel_weights = layers.Reshape((1, 1, input_shape[-1]))(x)

    # Multiply element-wise with the input feature map
    x = layers.multiply([inputs, channel_weights])

    # Flatten the result
    x = layers.Flatten()(x)

    # Fully connected layer to obtain final probability distribution
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to use the function
if __name__ == '__main__':
    model = dl_model()
    model.summary()

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))