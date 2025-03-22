# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications

def dl_model():
    """
    This function constructs a deep learning model for image classification using Keras Functional API.
    
    The model includes two sequential feature extraction layers, each consisting of a convolutional layer 
    followed by average pooling. This is followed by three additional convolutional layers and another 
    average pooling layer to reduce the dimensionality of the feature maps. After flattening the feature 
    maps, the model processes them through two fully connected layers, each accompanied by a dropout layer 
    to mitigate overfitting. Finally, the model outputs classification probabilities via a softmax layer 
    with 1,000 neurons.

    Args:
        None

    Returns:
        model (tf.keras.Model): The constructed deep learning model
    """
    
    # Input layer with shape 224x224x3
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Feature extraction layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    
    # Additional convolutional layers
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    
    # Flatten the feature maps
    x = layers.Flatten()(x)
    
    # Fully connected layers with dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer with softmax activation
    outputs = layers.Dense(1000, activation='softmax')(x)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model