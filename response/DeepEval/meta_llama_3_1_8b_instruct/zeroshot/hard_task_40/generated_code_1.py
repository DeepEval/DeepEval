from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, concatenate, Dense, Reshape, Dropout, Conv2D

def dl_model():
    """
    Function to create a deep learning model for image classification using the MNIST dataset.
    
    The model comprises two specialized blocks: 
    1. A block for feature extraction using pooling and convolutional layers.
    2. A block for classification using fully connected layers.
    
    Args:
        None
    
    Returns:
        The constructed model.
    """

    # Define input layer (28x28 images)
    inputs = Input(shape=(28, 28, 1))

    # First block: feature extraction using pooling and convolutional layers
    x = AveragePooling2D(pool_size=1, strides=1, padding='same')(inputs)  # 28x28
    x = AveragePooling2D(pool_size=2, strides=2, padding='same')(x)  # 14x14
    x = AveragePooling2D(pool_size=4, strides=4, padding='same')(x)  # 7x7
    x = Flatten()(x)
    x = concatenate([x, x, x, x])  # Concatenate 4 times the flattened output
    
    # Reshape to 4-dimensional tensor
    x = Reshape((784 * 4,))(x)
    
    # Second block: classification using fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Define parallel paths for multi-scale feature extraction
    path1 = Conv2D(32, 1, activation='relu', padding='same')(inputs)
    path1 = Dropout(0.2)(path1)
    
    path2 = Conv2D(32, 1, activation='relu', padding='same')(inputs)
    path2 = Conv2D(32, 3, activation='relu', padding='same')(path2)
    path2 = Conv2D(32, 3, activation='relu', padding='same')(path2)
    path2 = Dropout(0.2)(path2)
    
    path3 = Conv2D(32, 1, activation='relu', padding='same')(inputs)
    path3 = Conv2D(32, 3, activation='relu', padding='same')(path3)
    path3 = Dropout(0.2)(path3)
    
    path4 = AveragePooling2D(pool_size=2, strides=2, padding='same')(inputs)
    path4 = Conv2D(32, 1, activation='relu', padding='same')(path4)
    path4 = Dropout(0.2)(path4)
    
    # Concatenate outputs from all paths along the channel dimension
    outputs = concatenate([path1, path2, path3, path4], axis=-1)
    
    # Reshape and concatenate with the output from the first block
    outputs = Reshape((7, 7, 32 * 4))(outputs)
    outputs = concatenate([x, outputs])
    
    # Define output layer
    outputs = Dense(10, activation='softmax')(outputs)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model