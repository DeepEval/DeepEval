# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the input shape of the images
input_shape = (32, 32, 3)

# Define the dl_model function
def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model features a multi-branch convolutional architecture that processes input features at various scales using different convolutional kernel sizes and pooling.
    
    Parameters:
    None
    
    Returns:
    model: A Keras model instance.
    """
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape, name='inputs')
    
    # Define the first branch with 3x3 convolutions
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Conv2D(128, (3, 3), activation='relu')(branch1)
    
    # Define the second branch with 1x1 convolutions followed by two 3x3 convolutions
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = layers.Conv2D(128, (3, 3), activation='relu')(branch2)
    
    # Define the third branch with max pooling
    branch3 = layers.Conv2D(128, (3, 3), activation='relu')(inputs)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.Conv2D(256, (3, 3), activation='relu')(branch3)
    
    # Concatenate the outputs from the branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])
    
    # Apply dropout to the concatenated output
    concatenated = layers.Dropout(0.2)(concatenated)
    
    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)
    
    # Define the first fully connected layer
    x = layers.Dense(128, activation='relu')(flattened)
    
    # Define the second fully connected layer
    x = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)
    
    return model

# Create an instance of the deep learning model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()