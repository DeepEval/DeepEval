import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

def build_same_block(filters, kernel_size):
    """Builds a same block."""
    return tf.keras.Sequential(
        [
            layers.Conv2D(filters, kernel_size, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ]
    )

# Define the model
def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = inputs
    for _ in range(3):
        branch1 = build_same_block(64, 3)(branch1)

    # Branch 2
    branch2 = inputs
    for _ in range(3):
        branch2 = build_same_block(64, 5)(branch2)

    # Concatenate branches
    merged = layers.concatenate([branch1, branch2])

    # Fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(64, activation='relu')(merged)
    outputs = layers.Dense(10, activation='softmax')(merged)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)