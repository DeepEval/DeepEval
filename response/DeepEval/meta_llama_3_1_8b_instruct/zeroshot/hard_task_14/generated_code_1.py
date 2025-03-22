# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model comprises a main path and a branch path. In the main path, the input features undergo global average pooling 
    to extract global information from the feature map. The pooled output is then processed through two fully connected 
    layers to generate weights whose size is the same as the channels of the input layer. These weights are reshaped to 
    match the input layer's shape and multiplied element-wise with the original feature map. In the branch path, this path 
    connects to the input layer via a 3x3 convolution, adjusting the output feature map's channel to match that of the 
    input layer. The outputs from both paths are added together. Finally, the combined result is passed through three 
    fully connected layers to produce the final probability distribution for classification.
    """
    
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Main path
    main_path = layers.GlobalAveragePooling2D()(inputs)
    main_path = layers.Dense(128, activation='relu')(main_path)
    main_path = layers.Dense(128, activation='relu')(main_path)
    weights = layers.Reshape(target_shape=input_shape[1:], name='weights')(main_path)
    
    # Branch path
    branch_path = layers.Conv2D(3, (3, 3), activation='relu')(inputs)
    branch_path = layers.BatchNormalization()(branch_path)
    branch_path = layers.Conv2D(3, (3, 3), activation='relu')(branch_path)
    branch_path = layers.BatchNormalization()(branch_path)
    
    # Combine the main path and the branch path
    combined = layers.Add()([weights, branch_path])
    
    # Final classification layer
    x = layers.Flatten()(combined)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()