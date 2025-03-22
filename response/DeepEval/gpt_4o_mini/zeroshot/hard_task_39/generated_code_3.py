import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = layers.Input(shape=(28, 28, 1))

    # Block 1: Max Pooling Layers with different scales
    # 1x1 Max Pooling
    max_pool_1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flattened_1 = layers.Flatten()(max_pool_1)

    # 2x2 Max Pooling
    max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flattened_2 = layers.Flatten()(max_pool_2)

    # 4x4 Max Pooling
    max_pool_3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flattened_3 = layers.Flatten()(max_pool_3)

    # Concatenate flattened outputs
    concatenated = layers.concatenate([flattened_1, flattened_2, flattened_3])

    # Fully connected layer and reshape to 4D tensor
    fc_layer = layers.Dense(128, activation='relu')(concatenated)
    reshaped = layers.Reshape((4, 4, 8))(fc_layer)  # Reshape into a suitable 4D tensor

    # Block 2: Multiple branches
    # 1x1 Convolution branch
    conv_1x1 = layers.Conv2D(16, (1, 1), activation='relu')(reshaped)

    # 3x3 Convolution branch
    conv_3x3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(reshaped)

    # 5x5 Convolution branch
    conv_5x5 = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(reshaped)

    # 3x3 Max Pooling branch
    max_pool_branch = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)

    # Concatenate all branches
    concatenated_branches = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool_branch])

    # Flatten and output layer for classification
    flattened_output = layers.Flatten()(concatenated_branches)
    output_layer = layers.Dense(10, activation='softmax')(flattened_output)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of compiling the model
if __name__ == '__main__':
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Train the model (example with a small number of epochs)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)