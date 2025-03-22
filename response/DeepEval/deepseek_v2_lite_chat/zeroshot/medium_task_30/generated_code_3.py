import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

def dl_model():
    # Seed for reproducibility
    tf.random.set_seed(0)

    # Create a Sequential model
    model = Sequential([
        # Layer 1: Convolutional layer with 32 filters, 3x3 kernel, and ReLU activation
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        AveragePooling2D(pool_size=(1, 1)),  # Pooling with a stride of 1x1

        # Layer 2: Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),  # Pooling with a stride of 2x2

        # Layer 3: Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        AveragePooling2D(pool_size=(4, 4), strides=(4, 4)),  # Pooling with a stride of 4x4

        # Flatten the output to process through fully connected layers
        Flatten(),

        # Fully connected layer with 512 nodes
        Dense(512, activation='relu'),
        # Output layer with 10 nodes (for CIFAR-10's 10 classes)
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Construct the model
model = dl_model()