# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model has two feature extraction paths: 
    1. One path employs a 1x1 convolution.
    2. The second path consists of a sequence of convolutions: 
       1x1, followed by 1x7, and then 7x1.
    
    The outputs from these two paths will be concatenated, 
    and a 1x1 convolution will be applied to align the output dimensions with the input image's channel, 
    creating the output for the main path.
    
    Additionally, a branch will connect directly to the input, 
    merging the outputs of the main path and the branch through addition.
    
    Finally, the classification results will be produced through two fully connected layers.
    """
    
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Construct the model using Functional API
    inputs = keras.Input(shape=input_shape)
    
    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(32, 1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.MaxPooling2D(pool_size=(2, 2))(path1)
    path1 = layers.Conv2D(64, 1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(path1)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.MaxPooling2D(pool_size=(2, 2))(path1)
    
    # Path 2: sequence of convolutions
    path2 = layers.Conv2D(32, 1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Conv2D(32, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Conv2D(32, 1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.MaxPooling2D(pool_size=(2, 2))(path2)
    path2 = layers.Conv2D(64, 1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(path2)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.MaxPooling2D(pool_size=(2, 2))(path2)
    
    # Concatenate the outputs of the two paths
    concatenated = layers.Concatenate()([path1, path2])
    
    # Apply 1x1 convolution to align the output dimensions
    output = layers.Conv2D(64, 1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concatenated)
    output = layers.BatchNormalization()(output)
    
    # Direct connection to the input
    branch = inputs
    
    # Merge the outputs of the main path and the branch through addition
    merged = layers.Add()([output, branch])
    
    # Apply global average pooling
    x = layers.GlobalAveragePooling2D()(merged)
    
    # Add a fully connected layer
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    # Add the final fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model