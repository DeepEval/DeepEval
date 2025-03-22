import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# First block
def first_block(inputs, filters1, filters2, filters3):
    x = Conv2D(filters=filters1, kernel_size=1, activation='relu')(inputs)
    x = Conv2D(filters=filters1, kernel_size=3, activation='relu')(x)
    x = Conv2D(filters=filters1, kernel_size=5, activation='relu')(x)
    x = Conv2D(filters=filters2, kernel_size=1, activation='relu')(x)
    x = Conv2D(filters=filters2, kernel_size=3, activation='relu')(x)
    x = Conv2D(filters=filters2, kernel_size=5, activation='relu')(x)
    x = Conv2D(filters=filters3, kernel_size=1, activation='relu')(x)
    x = Conv2D(filters=filters3, kernel_size=3, activation='relu')(x)
    x = Conv2D(filters=filters3, kernel_size=5, activation='relu')(x)
    return x

# Second block
def second_block(inputs):
    x = Conv2D(64, 3, activation='relu')(inputs)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = AveragePooling2D(pool_size=3, strides=2, padding='same')(x)
    return x

# Concatenation and final classification
def model(input_shape):
    inputs = Input(shape=input_shape)
    
    # First block
    x = first_block(inputs, 16, 32, 64)
    
    # Second block
    x = second_block(x)
    
    # Global average pooling and final classification
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Create the model
model = model(input_shape)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Save the model
model.save('cifar10_model.h5')