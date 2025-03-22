# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the Functional API of Keras.
    The model consists of two blocks: Block 1 with three parallel paths and Block 2 with multiple branch connections.
    """
    
    # Define the input layer with shape (28, 28, 1) for the MNIST dataset
    inputs = keras.Input(shape=(28, 28, 1))
    
    # Block 1
    block1_path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    block1_path1 = layers.Flatten()(block1_path1)
    block1_path1 = layers.Dropout(0.2)(block1_path1)
    
    block1_path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    block1_path2 = layers.Flatten()(block1_path2)
    block1_path2 = layers.Dropout(0.2)(block1_path2)
    
    block1_path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    block1_path3 = layers.Flatten()(block1_path3)
    block1_path3 = layers.Dropout(0.2)(block1_path3)
    
    # Concatenate the outputs of the three paths
    block1_output = layers.Concatenate()([block1_path1, block1_path2, block1_path3])
    
    # Reshape the output to 4-dimensional tensor format
    block1_output = layers.Reshape(target_shape=(1, 1, 3*64))(block1_output)
    
    # Fully connected layer to transform the output of block 1
    block1_output = layers.Flatten()(block1_output)
    block1_output = layers.Dense(64, activation='relu')(block1_output)
    
    # Block 2
    block2_branch1 = layers.Conv2D(32, (1, 1), activation='relu')(block1_output)
    block2_branch1 = layers.Flatten()(block2_branch1)
    
    block2_branch2 = layers.Conv2D(32, (3, 3), activation='relu')(block1_output)
    block2_branch2 = layers.Flatten()(block2_branch2)
    
    block2_branch3 = layers.Conv2D(32, (1, 1), activation='relu')(block1_output)
    block2_branch3 = layers.Conv2D(32, (3, 3), activation='relu')(block2_branch3)
    block2_branch3 = layers.Flatten()(block2_branch3)
    
    block2_branch4 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(block1_output)
    block2_branch4 = layers.Conv2D(32, (1, 1), activation='relu')(block2_branch4)
    block2_branch4 = layers.Flatten()(block2_branch4)
    
    # Concatenate the outputs of the four branches
    block2_output = layers.Concatenate()([block2_branch1, block2_branch2, block2_branch3, block2_branch4])
    
    # Two fully connected layers for classification
    block2_output = layers.Dense(64, activation='relu')(block2_output)
    outputs = layers.Dense(10, activation='softmax')(block2_output)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model