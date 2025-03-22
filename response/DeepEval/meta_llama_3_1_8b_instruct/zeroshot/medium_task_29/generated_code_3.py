# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function creates a deep learning model using Keras' Functional API for image classification on the CIFAR-10 dataset.
    
    The model consists of three max pooling layers with varying window sizes (1x1, 2x2, and 4x4), followed by flattening and 
    concatenation of the output feature vectors. Finally, two fully connected layers are used to produce classification results 
    for the 10 classes in the CIFAR-10 dataset.
    """
    
    # Define the input layer with a shape of (32, 32, 3) for CIFAR-10 images
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Convolutional layer with 32 filters, kernel size of 3, and ReLU activation
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # First max pooling layer with a window size of 1x1 and stride equal to the size
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    
    # Convolutional layer with 64 filters, kernel size of 3, and ReLU activation
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    
    # Second max pooling layer with a window size of 2x2 and stride equal to the size
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Convolutional layer with 128 filters, kernel size of 3, and ReLU activation
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    
    # Third max pooling layer with a window size of 4x4 and stride equal to the size
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    
    # Flatten the output of the max pooling layers into a one-dimensional vector
    x = layers.Flatten()(x)
    
    # Concatenate the feature vectors from each pooling layer
    x = layers.Concatenate()([layers.Flatten()(layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)),
                              layers.Flatten()(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)),
                              layers.Flatten()(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x))])
    
    # First fully connected layer with 128 units, ReLU activation, and dropout
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Second fully connected layer with 10 units, softmax activation, and dropout
    output_layer = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()