from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Conv2D, concatenate
from keras.models import Model

def dl_model():
    # Input layer for the MNIST dataset (28x28 grayscale images)
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with average pooling
    # Path 1: Average pooling with 1x1 window
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flat1 = Flatten()(pool1)
    drop1 = Dropout(0.5)(flat1)
    
    # Path 2: Average pooling with 2x2 window
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flat2 = Flatten()(pool2)
    drop2 = Dropout(0.5)(flat2)
    
    # Path 3: Average pooling with 4x4 window
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat3 = Flatten()(pool3)
    drop3 = Dropout(0.5)(flat3)
    
    # Concatenate results from all paths in Block 1
    block1_output = concatenate([drop1, drop2, drop3])
    
    # Fully connected layer after Block 1
    fc1 = Dense(128, activation='relu')(block1_output)
    
    # Reshape operation to form a 4-dimensional tensor suitable for Block 2
    reshaped = Reshape((8, 8, 2))(fc1)
    
    # Block 2: Multiple branches for feature extraction
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    
    # Branch 2: 1x1 followed by 3x3 Convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    
    # Branch 3: 1x1, 3x3, then another 3x3 Convolution
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    
    # Branch 4: Average pooling followed by 1x1 Convolution
    branch4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped)
    branch4 = Conv2D(32, (1, 1), activation='relu', padding='same')(branch4)
    
    # Concatenate outputs from all branches in Block 2
    block2_output = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten before final fully connected layers
    flat_final = Flatten()(block2_output)
    
    # Final classification layers
    fc2 = Dense(128, activation='relu')(flat_final)
    output_layer = Dense(10, activation='softmax')(fc2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model