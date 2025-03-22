import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution for dimensionality reduction
    branch1 = layers.Conv2D(64, (1, 1), padding='same')(inputs)

    # Branch 2: 1x1 and 3x3 convolutions
    branch2 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    branch2 = layers.Conv2D(128, (3, 3), padding='same')(branch2)

    # Branch 3: 1x1 and 5x5 convolutions
    branch3 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    branch3 = layers.Conv2D(128, (5, 5), padding='same')(branch3)

    # Branch 4: 3x3 max pooling and 1x1 convolution
    branch4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputs)
    branch4 = layers.Conv2D(64, (1, 1), padding='same')(branch4)

    # Concatenate outputs from all branches
    concat = layers.concatenate([branch1, branch2, branch3, branch4])

    # Flatten the concatenated features
    flatten = layers.Flatten()(concat)

    # Fully connected layers for classification
    fc1 = layers.Dense(512, activation='relu')(flatten)
    fc2 = layers.Dense(10, activation='softmax')(fc1)

    # Model definition
    model = models.Model(inputs=inputs, outputs=fc2)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert to float32 and normalize
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)