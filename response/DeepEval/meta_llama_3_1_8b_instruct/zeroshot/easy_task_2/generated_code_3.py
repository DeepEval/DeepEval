# Import necessary packages
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    """
    Creates a deep learning model for image classification.
    
    The model consists of two sequential feature extraction layers, each 
    consisting of a convolutional layer followed by average pooling. This is 
    followed by three additional convolutional layers and another average 
    pooling layer to reduce the dimensionality of the feature maps. After 
    flattening the feature maps, the model processes them through two fully 
    connected layers, each accompanied by a dropout layer to mitigate 
    overfitting. Finally, the model outputs classification probabilities via 
    a softmax layer with 1,000 neurons.
    """
    
    # Define input shape
    input_shape = (224, 224, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Sequential feature extraction layers
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = AveragePooling2D((2, 2), name='pool1')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = AveragePooling2D((2, 2), name='pool2')(x)
    
    # Additional convolutional layers
    x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv4')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv5')(x)
    
    # Average pooling layer
    x = AveragePooling2D((2, 2), name='pool3')(x)
    
    # Flatten the feature maps
    x = Flatten(name='flatten')(x)
    
    # Fully connected layers with dropout
    x = Dense(128, activation='relu', name='dense1')(x)
    x = Dropout(0.2, name='dropout1')(x)
    
    x = Dense(64, activation='relu', name='dense2')(x)
    x = Dropout(0.2, name='dropout2')(x)
    
    # Output layer with softmax activation
    outputs = Dense(1000, activation='softmax', name='output')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model