# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    A deep learning model for image classification using the MNIST dataset.
    
    The model comprises two specialized blocks:
    1. The first block processes the input through three average pooling layers with 
       pooling windows and strides of 1x1, 2x2, and 4x4, respectively. The outputs from 
       these pooling layers are flattened into one-dimensional vectors and concatenated 
       to create a vector. A fully connected layer, followed by a reshape operation, 
       converts this output into a 4-dimensional tensor suitable for the second block.
    2. The second block consists of four parallel paths, each designed to extract 
       multi-scale features using different convolutional kernels. Path 1 employs a 
       1x1 convolution for quick feature extraction. Path 2 features two 3x3 convolutions 
       stacked after a 1x1 convolution. Path 3 includes a single 3x3 convolution following 
       a 1x1 convolution. While Path 4 combines average pooling with a 1x1 convolution for 
       feature compression and extraction. Each path ends with a dropout layer to mitigate 
       overfitting. The outputs from all paths are then concatenated along the channel 
       dimension. Finally, the model outputs the classification result through two fully 
       connected layers.
    """
    
    # Input layer with shape (28, 28, 1) representing the MNIST images
    input_layer = keras.Input(shape=(28, 28, 1))
    
    # First block: three average pooling layers with pooling windows and strides of 1x1, 2x2, and 4x4
    pool1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(pool1)
    pool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(pool2)
    
    # Flatten and concatenate the outputs from the pooling layers
    flatten1 = layers.Flatten()(pool1)
    flatten2 = layers.Flatten()(pool2)
    flatten3 = layers.Flatten()(pool3)
    concatenated = layers.Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer with 128 units
    fc = layers.Dense(128, activation='relu')(concatenated)
    
    # Reshape operation to convert the output into a 4-dimensional tensor
    reshaped = layers.Reshape((1, 1, 128))(fc)
    
    # Second block: four parallel paths for multi-scale feature extraction
    path1 = layers.Conv2D(64, (1, 1), activation='relu')(reshaped)
    path2 = layers.Conv2D(64, (1, 1), activation='relu')(reshaped)
    path2 = layers.Conv2D(64, (3, 3), activation='relu')(path2)
    path2 = layers.Conv2D(64, (3, 3), activation='relu')(path2)
    path3 = layers.Conv2D(64, (1, 1), activation='relu')(reshaped)
    path3 = layers.Conv2D(64, (3, 3), activation='relu')(path3)
    path4 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(reshaped)
    path4 = layers.Conv2D(64, (1, 1), activation='relu')(path4)
    
    # Dropout layer to mitigate overfitting
    path1 = layers.Dropout(0.2)(path1)
    path2 = layers.Dropout(0.2)(path2)
    path3 = layers.Dropout(0.2)(path3)
    path4 = layers.Dropout(0.2)(path4)
    
    # Concatenate the outputs from all paths along the channel dimension
    concatenated_paths = layers.Concatenate()([path1, path2, path3, path4])
    
    # Flatten and fully connected layer for classification
    flatten = layers.Flatten()(concatenated_paths)
    fc = layers.Dense(128, activation='relu')(flatten)
    
    # Output layer with softmax activation for classification
    output_layer = layers.Dense(10, activation='softmax')(fc)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model