import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block: Three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten each pooled output
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer
    fc1 = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((1, 1, 128))(fc1)  # Reshape to 4D tensor for the second block

    # Second block: Four parallel paths
    # Path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path1 = Dropout(0.5)(path1)

    # Path 2: Two 3x3 convolutions stacked after a 1x1 convolution
    path2 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: Single 3x3 convolution after 1x1 convolution
    path3 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path3 = Dropout(0.5)(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D(pool_size=(2, 2))(reshaped)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)
    path4 = Dropout(0.5)(path4)

    # Concatenate the outputs from all paths
    concatenated_paths = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated outputs
    flattened_paths = Flatten()(concatenated_paths)

    # Output layers
    fc2 = Dense(64, activation='relu')(flattened_paths)
    output_layer = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of compiling the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)