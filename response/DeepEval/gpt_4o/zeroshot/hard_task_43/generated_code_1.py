from keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

def dl_model():
    # Define the input layer for the MNIST dataset (28x28 grayscale images)
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with AveragePooling
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    path1 = Flatten()(path1)
    
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    path2 = Flatten()(path2)
    
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(inputs)
    path3 = Flatten()(path3)
    
    # Concatenate outputs of the three paths
    concatenated = Concatenate()([path1, path2, path3])
    
    # Fully connected layer between Block 1 and Block 2
    fc1 = Dense(512, activation='relu')(concatenated)
    
    # Reshape for Block 2
    reshaped = Reshape((8, 8, 8))(fc1)  # Assuming suitable dimensions for reshaping
    
    # Block 2: Three branches for feature extraction
    # Branch 1: 1x1 convolution followed by 3x3 convolution
    branch1 = Conv2D(16, (1, 1), padding='same', activation='relu')(reshaped)
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch1)
    
    # Branch 2: 1x1, 1x7, 7x1 convolutions, followed by 3x3 convolution
    branch2 = Conv2D(16, (1, 1), padding='same', activation='relu')(reshaped)
    branch2 = Conv2D(16, (1, 7), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(16, (7, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: Average pooling
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    
    # Concatenate outputs of the three branches
    concatenated_block2 = Concatenate()([branch1, branch2, branch3])
    
    # Global average pooling layer
    gap = GlobalAveragePooling2D()(concatenated_block2)
    
    # Fully connected layers for classification
    fc2 = Dense(128, activation='relu')(gap)
    outputs = Dense(10, activation='softmax')(fc2)  # MNIST has 10 classes
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()