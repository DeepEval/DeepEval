from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the Keras Functional API.
    
    The model extracts multi-scale and multi-directional features from the input through four parallel branches.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model.
    """
    
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # Path 2: Average pooling + 1x1 convolution
    path2 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    path2 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(path2)
    
    # Path 3: 1x1 convolution + 1x3 and 3x1 convolutions
    path3 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)
    path3 = layers.Conv2D(32, kernel_size=(1, 3), activation='relu')(path3)
    path3 = layers.Conv2D(32, kernel_size=(3, 1), activation='relu')(path3)
    
    # Path 4: 1x1 convolution + 3x3 convolution + 1x3 and 3x1 convolutions
    path4 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)
    path4 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path4)
    path4 = layers.Conv2D(32, kernel_size=(1, 3), activation='relu')(path4)
    path4 = layers.Conv2D(32, kernel_size=(3, 1), activation='relu')(path4)
    
    # Concatenate the outputs of the four paths
    concatenated = layers.Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)
    
    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model