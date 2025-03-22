import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    # Third convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Concatenate the output of each convolutional layer
    x = Concatenate()([x, input_layer])
    
    # Pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer 1
    x = Dense(512, activation='relu')(x)
    # Fully connected layer 2
    x = Dense(10, activation='softmax')(x)  # Assuming binary classification, change to 'softmax' for multi-class
    
    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()