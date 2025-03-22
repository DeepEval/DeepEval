# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    Create a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. The main path splits the input into three groups by 
    encapsulating tf.split within a Lambda layer, each undergoing feature extraction with convolutional 
    layers of different kernel sizes (1x1, 3x3, and 5x5). The outputs from these three groups are then 
    concatenated. The branch path processes the input with a 1x1 convolutional layer to align the number 
    of output channels with those of the main path. The outputs of the main and branch paths are combined 
    through addition to create fused features. Finally, the model performs classification using two fully 
    connected layers.
    
    Returns:
        model (tf.keras.Model): The constructed deep learning model.
    """
    
    # Define the input shape
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups using a Lambda layer
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Define the main path
    main_path = []
    for i in range(3):
        # Create a convolutional layer with different kernel sizes
        x = layers.Conv2D(32, kernel_size=1 if i == 0 else 3 if i == 1 else 5, activation='relu')(split_layer[i])
        main_path.append(x)
    
    # Concatenate the outputs from the main path
    concatenated = layers.Concatenate()(main_path)
    
    # Define the branch path
    branch_path = layers.Conv2D(32, kernel_size=1, activation='relu')(inputs)
    
    # Combine the outputs of the main and branch paths
    x = layers.Add()([concatenated, branch_path])
    
    # Define the classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()