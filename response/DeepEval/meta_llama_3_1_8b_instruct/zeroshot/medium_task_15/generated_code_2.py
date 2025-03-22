import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture using Functional API
    inputs = keras.Input(shape=input_shape)

    # Convolutional layer to extract initial features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Batch normalization
    x = layers.BatchNormalization()(x)
    
    # Convolutional layer to further extract features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Batch normalization
    x = layers.BatchNormalization()(x)
    
    # Global average pooling to compress feature maps
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer with 32 units and ReLU activation
    x = layers.Dense(32, activation='relu')(x)
    
    # Reshape to match the size of the initial feature
    x = layers.Reshape((32, 32, 1))(x)
    
    # Multiply with the initial features to generate weighted feature maps
    x = layers.Multiply()([x, inputs])
    
    # Concatenate with the input layer
    x = layers.Concatenate()([x, inputs])
    
    # 1x1 convolution to reduce dimensionality and downsample features
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    
    # Average pooling to further downsample features
    x = layers.AveragePooling2D((2, 2))(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Single fully connected layer to output classification results
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the model
model = dl_model()
model.summary()