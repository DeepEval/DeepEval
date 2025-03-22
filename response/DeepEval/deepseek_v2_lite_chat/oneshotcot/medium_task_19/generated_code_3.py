import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)
    
    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(branch3)
    
    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch4)
    
    # Concatenate the outputs of the four branches
    concatenated = Concatenate()(list(branch1, branch2, branch3, branch4))
    
    # Batch normalization
    bn = BatchNormalization()(concatenated)
    
    # Flatten the features
    flattened = Flatten()(bn)
    
    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(fc2)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model