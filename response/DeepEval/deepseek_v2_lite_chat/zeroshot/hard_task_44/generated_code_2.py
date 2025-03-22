import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)  # One-third of the original image size for CIFAR-10

# Function to create the first block of the model
def create_first_block(input_tensor, filters):
    x = Conv2D(filters, (1, 1), activation='relu')(input_tensor)
    x = Conv2D(filters, (3, 3), activation='relu')(x)
    x = Conv2D(filters, (5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.split(x, 3, axis=-1)  # Split along the channel dimension
    return x

# Function to create the second block of the model
def create_second_block(x_list):
    # Define the branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(x_list[0])
    branch2 = Conv2D(64, (1, 1), activation='relu')(x_list[1])
    branch3 = Conv2D(64, (3, 3), activation='relu')(x_list[2])
    branch4 = Conv2D(64, (1, 1), activation='relu')(x_list[3])
    branch5 = MaxPooling2D(pool_size=(3, 3))(x_list[4])

    # Concatenate the branch outputs
    x = concatenate([branch1, branch2, branch3, branch4, branch5])
    x = Conv2D(128, (1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # Assuming the correct number of classes

    return output

# Initialize the input tensor
inputs = Input(shape=input_shape)

# Create the first block
x = create_first_block(inputs, 64)

# Create the second block
output = create_second_block(x)

# Define the model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Return the model
return model