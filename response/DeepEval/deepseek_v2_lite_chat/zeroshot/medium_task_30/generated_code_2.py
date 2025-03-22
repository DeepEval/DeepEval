import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Concatenate

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)  # Input images are 32x32

    # Create the model
    inputs = Input(shape=input_shape)

    # First convolutional layer with average pooling
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)

    # Second convolutional layer with average pooling
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Third convolutional layer with average pooling
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)

    # Flatten and concatenate the outputs
    x = Flatten()(x)
    x = tf.concat([x, x, x], axis=-1)  # Concatenate the outputs of each pooling layer

    # Flatten and process through two fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # Output layer with 10 classes

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()