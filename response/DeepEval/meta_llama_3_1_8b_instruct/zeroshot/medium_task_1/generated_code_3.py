# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins with two convolutional layers followed by a max-pooling layer to extract features.
    The output features are directly added with the input layer. Finally, these features are flattened 
    and processed through two fully connected layers to produce a probability distribution across the 10 classes.
    
    Returns:
        model (Model): The constructed deep learning model.
    """
    
    # Define the input shape of the images
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional layer 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Convolutional layer 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Max-pooling layer
    x = MaxPooling2D((2, 2))(x)
    
    # Add the input layer to the output features (shortcut connection)
    x = Add()([inputs, x])
    
    # Flatten the output features
    x = Flatten()(x)
    
    # Dense layer 1
    x = Dense(128, activation='relu')(x)
    
    # Dense layer 2 (output layer)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model