import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # 1x1 Convolution branch
    conv_1x1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    
    # 3x3 Convolution branch
    conv_3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # 5x5 Convolution branch
    conv_5x5 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)

    # 3x3 Max Pooling branch
    max_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate all branches
    concatenated = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool])

    # Flatten the concatenated output
    flatten = layers.Flatten()(concatenated)

    # Fully connected layers
    dense_1 = layers.Dense(128, activation='relu')(flatten)
    dense_2 = layers.Dense(64, activation='relu')(dense_1)

    # Output layer with softmax activation for classification
    output_layer = layers.Dense(10, activation='softmax')(dense_2)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize the images to [0, 1] range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construct the model
model = dl_model()
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()