# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. In the main path, the input is 
    splitted into three groups along the channel by encapsulating tf.split within Lambda layer, 
    each undergoing multi-scale feature extraction with separable convolutional layers of varying 
    kernel sizes (1x1, 3x3, and 5x5). The outputs from these groups are concatenated to produce 
    the output of the main path. The branch path applies a 1x1 convolutional layer to the input 
    to align the number of output channels with those of the main path. The outputs from both paths 
    are then fused through addition. Finally, the combined output is flattened into a one-dimensional 
    vector and passed through two fully connected layers for a 10-class classification task.
    """
    
    # Define the input shape of the model, which is 32x32x3 for CIFAR-10
    input_shape = (32, 32, 3)
    
    # Create the main model
    main_model = keras.Sequential()
    
    # Split the input into three groups along the channel using Lambda layer
    main_model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1)))
    
    # Apply multi-scale feature extraction with separable convolutional layers of varying kernel sizes
    main_model.add(layers.SeparableConv2D(32, (1, 1), activation='relu', input_shape=input_shape))
    main_model.add(layers.SeparableConv2D(32, (3, 3), activation='relu'))
    main_model.add(layers.SeparableConv2D(64, (5, 5), activation='relu'))
    
    # Concatenate the outputs from the three groups
    main_model.add(layers.Concatenate(axis=-1))
    
    # Apply a global average pooling layer to reduce the spatial dimensions
    main_model.add(layers.GlobalAveragePooling2D())
    
    # Apply a dense layer to reduce the number of channels
    main_model.add(layers.Dense(64, activation='relu'))
    
    # Create the branch model
    branch_model = keras.Sequential()
    
    # Apply a 1x1 convolutional layer to the input to align the number of output channels with those of the main path
    branch_model.add(layers.Conv2D(64, (1, 1), activation='relu', input_shape=input_shape))
    
    # Apply a global average pooling layer to reduce the spatial dimensions
    branch_model.add(layers.GlobalAveragePooling2D())
    
    # Apply a dense layer to reduce the number of channels
    branch_model.add(layers.Dense(64, activation='relu'))
    
    # Fuse the outputs from both paths through addition
    combined_model = keras.Sequential()
    combined_model.add(layers.Concatenate(axis=-1))
    combined_model.add(layers.Add())
    
    # Concatenate the main model and the branch model
    combined_model.add(main_model)
    combined_model.add(branch_model)
    
    # Flatten the combined output into a one-dimensional vector
    combined_model.add(layers.Flatten())
    
    # Apply two fully connected layers for a 10-class classification task
    combined_model.add(layers.Dense(64, activation='relu'))
    combined_model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Return the constructed model
    return combined_model

# Test the function
model = dl_model()
print(model.summary())