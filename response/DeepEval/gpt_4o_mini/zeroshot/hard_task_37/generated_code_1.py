import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def create_block(input_tensor):
    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    output1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Second path
    output2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    
    # Addition of both outputs
    added = layers.add([output1, output2])
    
    return added

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    inputs = layers.Input(shape=input_shape)
    
    # First parallel branch
    block1_output = create_block(inputs)
    
    # Second parallel branch
    block2_output = create_block(inputs)
    
    # Concatenate the outputs of both blocks
    concatenated = layers.concatenate([block1_output, block2_output])
    
    # Flatten the output
    flattened = layers.Flatten()(concatenated)
    
    # Fully connected layer
    dense_output = layers.Dense(128, activation='relu')(flattened)
    
    # Output layer for classification (10 classes for MNIST)
    outputs = layers.Dense(10, activation='softmax')(dense_output)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()