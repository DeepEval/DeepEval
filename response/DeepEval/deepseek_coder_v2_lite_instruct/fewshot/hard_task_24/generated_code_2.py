import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, UpSampling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 1: Local features with 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Branch 2: Sequential operations including max pooling, 3x3 convolutional layer, and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    
    # Branch 3: Sequential operations including max pooling, 3x3 convolutional layer, and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    
    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Further processing with another 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Flatten the output for fully connected layers
    flattened = Flatten()(conv2)
    
    # Three fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model