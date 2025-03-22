from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    The model consists of two sequential blocks. In Block 1, global average pooling generates weights 
    that pass through two fully connected layers with the same channel as the input layer. These weights 
    are reshaped to match the input's shape and multiplied with the input to produce the weighted feature output.
    Block 2 extracts deep features using two 3x3 convolutional layers followed by a max pooling layer. A branch 
    from Block 1 connects directly to the output of Block 2. The outputs from the main path and the branch are 
    then fused through addition. Finally, the combined output is classified using two fully connected layers.
    """
    
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define Block 1
    inputs_block1 = keras.Input(shape=input_shape)
    
    # Global average pooling
    avg_pooling = layers.GlobalAveragePooling2D()(inputs_block1)
    
    # Two fully connected layers
    fc1 = layers.Dense(10, activation='relu')(avg_pooling)
    fc2 = layers.Dense(10, activation='relu')(fc1)
    
    # Reshape the weights to match the input's shape
    weights = layers.Reshape(input_shape)(fc2)
    
    # Multiply the weights with the input to produce the weighted feature output
    weighted_output_block1 = layers.Multiply()([inputs_block1, weights])
    
    # Define Block 2
    inputs_block2 = weighted_output_block1
    
    # Convolutional and max pooling layers
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs_block2)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(conv1)
    max_pooling = layers.MaxPooling2D((2, 2))(conv2)
    
    # Add a branch from Block 1 to the output of Block 2
    concatenated = layers.Concatenate()([max_pooling, fc2])
    
    # Two fully connected layers for classification
    fc3 = layers.Dense(64, activation='relu')(concatenated)
    outputs = layers.Dense(10, activation='softmax')(fc3)
    
    # Define the model
    model = keras.Model(inputs=inputs_block1, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()