import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the size of the input image
input_shape = (32, 32, 3)


def first_block(inputs):
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    split_1 = tf.split(x, num_or_size_splits=3, axis=-1)
    
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(split_1[0])
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    split_2 = tf.split(x, num_or_size_splits=3, axis=-1)
    
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(split_2[0])
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    split_3 = tf.split(x, num_or_size_splits=3, axis=-1)
    
    x = tf.concat(split_3[1:], axis=-1)  # Concatenate after splitting
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    
    return x

def second_block(x):
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    return x

def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First block
    x = first_block(inputs)
    
    # Second block
    x = second_block(x)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()