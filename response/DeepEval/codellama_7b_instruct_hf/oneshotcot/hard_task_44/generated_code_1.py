import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Define the input shape
input_shape = (32, 32, 3)

# Define the first block
def block_1(input_tensor):
    # Split the input into three groups along the channel dimension
    groups = tf.split(input_tensor, num_or_size_splits=3, axis=3)

    # Apply convolutional layers with varying kernel sizes to each group
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(groups[0])
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(groups[1])
    conv3 = layers.Conv2D(32, (5, 5), activation='relu')(groups[2])

    # Apply dropout to reduce overfitting
    dropout = layers.Dropout(0.2)(conv1)

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv1, conv2, conv3])

    # Return the concatenated output
    return concatenated

# Define the second block
def block_2(input_tensor):
    # Split the input into four branches
    branches = tf.split(input_tensor, num_or_size_splits=4, axis=3)

    # Apply convolutional layers with varying kernel sizes to each branch
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(branches[0])
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(branches[1])
    conv3 = layers.Conv2D(32, (5, 5), activation='relu')(branches[2])
    max_pooling = layers.MaxPooling2D((1, 1))(branches[3])

    # Concatenate the outputs from all branches
    concatenated = layers.Concatenate()([conv1, conv2, conv3, max_pooling])

    # Return the concatenated output
    return concatenated

# Define the model
model = keras.Sequential([
    # First block
    layers.Lambda(block_1),
    # Second block
    layers.Lambda(block_2),
    # Flatten the output
    layers.Flatten(),
    # Dense layer with 128 units and ReLU activation
    layers.Dense(128, activation='relu'),
    # Dense layer with 64 units and ReLU activation
    layers.Dense(64, activation='relu'),
    # Dense layer with 10 units and softmax activation
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))