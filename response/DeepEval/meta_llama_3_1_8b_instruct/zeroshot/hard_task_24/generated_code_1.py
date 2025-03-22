# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function generates a deep learning model for image classification.
    
    The model is designed to classify images from the CIFAR-10 dataset. It consists of:
    1. An initial 1x1 convolutional layer to reduce the number of channels.
    2. Three separate branches:
        * Branch 1: Extracts local features through a 3x3 convolutional layer.
        * Branch 2 and Branch 3: Sequentially pass through a max pooling layer for downsampling, 
                                a 3x3 convolutional layer, and then an upsampling layer to restore to the original size.
    3. The outputs of all branches are fused together through concatenation and passed through another 1x1 convolutional layer.
    4. Finally, the output is passed through three fully connected layers to produce a 10-class classification result.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model for image classification.
    """

    # Input layer (32x32x3) to the model
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1: Extract local features through a 3x3 convolutional layer
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Branch 2 and Branch 3: Sequentially pass through a max pooling layer for downsampling, 
    # a 3x3 convolutional layer, and then an upsampling layer to restore to the original size
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = layers.UpSampling2D((2, 2))(branch2)
    
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.UpSampling2D((2, 2))(branch3)
    
    # Fuse the outputs of all branches through concatenation
    fused_branches = layers.Concatenate()([branch1, branch2, branch3])
    
    # Pass the fused output through another 1x1 convolutional layer
    fused_branches = layers.Conv2D(32, (1, 1), activation='relu')(fused_branches)
    
    # Flatten the output
    flat_output = layers.Flatten()(fused_branches)
    
    # Pass the flattened output through three fully connected layers
    outputs = layers.Dense(64, activation='relu')(flat_output)
    outputs = layers.Dense(32, activation='relu')(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model