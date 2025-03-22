import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the [0, 1] range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the datasets according to the model's expected input shape
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Create the first convolutional block
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(Input(shape=input_shape))
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Create the second convolutional block
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Flatten the output from the second block
    x = Flatten()(x)
    
    # Create the first fully connected layer
    x = Dense(units=128, activation='relu')(x)
    
    # Create the second fully connected layer
    x = Dense(units=64, activation='relu')(x)
    
    # Create the output layer
    output = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=Input(shape=input_shape), outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Print a summary of the model
model.summary()