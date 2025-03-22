# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    """
    This function creates a deep learning model for image classification using the MNIST dataset.
    
    The model features a specialized block designed to capture local features through a 3x3 convolutional layer, 
    followed by two 1x1 convolutional layers for additional processing. An average pooling layer is then applied 
    to downsample the feature maps, and a dropout layer is included to mitigate overfitting. After two consecutive 
    the blocks, a global average pooling layer converts the high-dimensional feature maps into a one-dimensional vector.
    This is followed by a flattening layer and a fully connected layer that produces the final classification output.
    """
    
    # Define the input layer (28x28 images)
    input_layer = Input(shape=(28, 28, 1))
    
    # Create the first block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Create the second block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Apply global average pooling
    x = AveragePooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Create the final output layer
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model