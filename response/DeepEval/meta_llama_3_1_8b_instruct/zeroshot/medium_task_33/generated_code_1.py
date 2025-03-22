# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the deep learning model function
def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = keras.Sequential([
        # Lambda layer to split the input image into three channel groups
        layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_shape),
        
        # Define the feature extraction block for each channel group
        layers.Lambda(lambda x: [
            tf.nn.separable_conv2d(x[0], [1, 3, 3, 32], [1, 1, 1, 1], padding='same', stride=1),
            tf.nn.separable_conv2d(x[1], [1, 3, 3, 64], [1, 1, 1, 1], padding='same', stride=1),
            tf.nn.separable_conv2d(x[2], [1, 3, 3, 96], [1, 1, 1, 1], padding='same', stride=1),
        ]),
        
        # Concatenate the outputs from the three channel groups
        layers.Concatenate(axis=-1),
        
        # Define the fully connected layer block
        layers.Lambda(lambda x: [
            tf.reshape(x, [-1, 1, 1, 96 * 3]),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x),
            layers.Flatten()(x)
        ]),
        
        # Define the final fully connected layer to produce the final probability outputs
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Call the dl_model function to get the constructed model
constructed_model = dl_model()
print(constructed_model.summary())