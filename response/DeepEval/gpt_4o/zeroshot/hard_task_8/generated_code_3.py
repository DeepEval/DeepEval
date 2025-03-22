import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 and grayscale

    inputs = Input(shape=input_shape)

    # Block 1
    # Primary path
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)

    # Branch path
    y = DepthwiseConv2D((3, 3), activation='relu', padding='same')(inputs)
    y = Conv2D(64, (1, 1), activation='relu', padding='same')(y)

    # Concatenate features from both paths
    concatenated = Concatenate(axis=-1)([x, y])

    # Block 2
    # Obtain shape of features from Block 1
    shape = concatenated.shape
    height, width, channels = shape[1], shape[2], shape[3]
    groups = 4
    channels_per_group = channels // groups

    # Reshape into (height, width, groups, channels_per_group)
    reshaped = Reshape((height, width, groups, channels_per_group))(concatenated)

    # Swap the third and fourth dimensions using permutation
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to original shape
    shuffled = Reshape((height, width, channels))(permuted)

    # Fully connected layer for classification
    flattened = Flatten()(shuffled)
    outputs = Dense(10, activation='softmax')(flattened)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == '__main__':
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create model
    model = dl_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")