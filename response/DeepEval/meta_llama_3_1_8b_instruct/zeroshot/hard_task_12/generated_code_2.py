from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the main path
    main_path = keras.Input(shape=input_shape)
    
    # Apply 1x1 convolutional layer for dimensionality reduction
    x = layers.Conv2D(32, (1, 1), activation='relu')(main_path)
    
    # Apply parallel convolutional layers with 1x1 and 3x3 kernels for feature extraction
    x1 = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x2 = layers.Conv2D(32, (3, 3), activation='relu')(x)
    
    # Concatenate the outputs of the two parallel convolutional layers
    x = layers.Concatenate()([x1, x2])
    
    # Apply 3x3 convolutional layer in the branch path to match the channel dimensions
    branch_path = layers.Conv2D(32, (3, 3), activation='relu')(main_path)
    
    # Combine the outputs from the main and branch paths using addition
    x = layers.Add()([x, branch_path])
    
    # Apply flattening layer followed by two fully connected layers for classification
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=main_path, outputs=x)
    
    return model