# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two sequential blocks: 
    - The first block has three convolutional layers followed by max pooling.
    - The second block has four convolutional layers followed by max pooling.
    - After feature extraction, the feature maps are flattened and fed into three fully connected layers, culminating in the classification output.
    
    Parameters:
    None
    
    Returns:
    model (Model): The constructed deep learning model.
    """
    
    # Define the input layer with shape (28, 28, 1) for MNIST dataset images
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1: Three convolutional layers followed by max pooling
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Block 2: Four convolutional layers followed by max pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # Three fully connected layers for classification output
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model