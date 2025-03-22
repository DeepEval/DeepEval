# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Activation
import tensorflow as tf

def dl_model():
    """
    This function creates a deep learning model for image classification.
    
    The model consists of two sequential feature extraction layers, 
    each consisting of a convolutional layer followed by average pooling.
    This is followed by three additional convolutional layers and another 
    average pooling layer to reduce the dimensionality of the feature maps.
    After flattening the feature maps, the model processes them through two 
    fully connected layers, each accompanied by a dropout layer to mitigate 
    overfitting. Finally, the model outputs classification probabilities via 
    a softmax layer with 1,000 neurons.
    
    Parameters:
    None
    
    Returns:
    model: The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (224, 224, 3)
    
    # Create the input layer
    input_layer = Input(shape=input_shape)
    
    # Feature extraction layers
    # Layer 1
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Layer 2
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Convolutional layers
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    
    # Average pooling layer
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # Fully connected layers with dropout
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    x = Dense(1000, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    
    return model