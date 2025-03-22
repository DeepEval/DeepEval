import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Convert class labels to binary vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# Define the input shape
input_shape = (32, 32, 3)

# Define the model
model = keras.Sequential([
    # Split the input into three groups along the last dimension
    layers.Lambda(lambda x: tf.split(x, 3, axis=-1)),

    # Feature extraction via depthwise separable convolutional layers
    layers.Conv2D(32, 1, activation='relu', input_shape=input_shape),
    layers.DepthwiseSeparableConv2D(32, 3, activation='relu'),
    layers.DepthwiseSeparableConv2D(32, 5, activation='relu'),

    # Concatenate the outputs of the three groups
    layers.Concatenate(),

    # Flatten the fused features
    layers.Flatten(),

    # Fully connected layer for classification
    layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')