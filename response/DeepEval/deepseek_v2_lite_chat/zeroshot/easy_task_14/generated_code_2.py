import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layer 1
    x = Dense(512, activation='relu')(x)
    
    # Fully connected layer 2
    x = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and return the model
model = dl_model()