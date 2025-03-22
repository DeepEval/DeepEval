# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by splitting the input into three groups along the channel dimension 
    by encapsulating tf.split within Lambda layer. Each group undergoes a sequence of 
    <1x1 convolution, 3x3 convolution> to extract deep features, followed by a dropout 
    layer for feature selection to mitigate overfitting. The outputs from the three groups 
    are then concatenated to form the main pathway. In parallel, a branch pathway processes 
    the input through a 1x1 convolution to match the output dimension of the main pathway. 
    The outputs from both pathways are combined using an addition operation. Finally, the 
    resulting output is passed through a fully connected layer to complete the classification process.
    
    Returns:
    model: The constructed deep learning model.
    """
    
    # Define the input shape for the model
    input_shape = (32, 32, 3)
    
    # Define the model
    inputs = keras.Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Process each group separately
    group1 = layers.Conv2D(32, 1, activation='relu')(split_input[0])
    group1 = layers.Conv2D(32, 3, activation='relu')(group1)
    group1 = layers.Dropout(0.2)(group1)
    
    group2 = layers.Conv2D(32, 1, activation='relu')(split_input[1])
    group2 = layers.Conv2D(32, 3, activation='relu')(group2)
    group2 = layers.Dropout(0.2)(group2)
    
    group3 = layers.Conv2D(32, 1, activation='relu')(split_input[2])
    group3 = layers.Conv2D(32, 3, activation='relu')(group3)
    group3 = layers.Dropout(0.2)(group3)
    
    # Concatenate the outputs from the three groups
    main_pathway = layers.Concatenate()([group1, group2, group3])
    
    # Define the branch pathway
    branch = layers.Conv2D(32, 1, activation='relu')(inputs)
    
    # Combine the outputs from both pathways
    combined = layers.Add()([main_pathway, branch])
    
    # Flatten the output
    flat = layers.Flatten()(combined)
    
    # Define the output layer
    outputs = layers.Dense(10, activation='softmax')(flat)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model