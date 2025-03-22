import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Function to create the model
def dl_model():
    # Input layer
    input_img = Input(shape=input_shape)
    
    # Convolutional layer
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layer 1
    x = Dense(256, activation='relu')(x)
    
    # Fully connected layer 2
    x = Dense(128, activation='relu')(x)
    
    # Output layer
    output = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_img, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))