import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1_output = input_layer

    for _ in range(3):  # Three paths for pooling
        # Path 1: 1x1 average pooling
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(block1_output)
        
        # Path 2: 2x2 average pooling
        block1_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block1_output)
        
        # Path 3: 4x4 average pooling
        block1_output = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(block1_output)
    
    # Flatten the combined output and concatenate
    flat_output = Flatten()(block1_output)
    concated_output = Concatenate()([flat_output, block1_output, block1_output, block1_output])
    
    # Add batch normalization and dense layers before Block 2
    bath_norm = BatchNormalization()(concated_output)
    dense_layer_1 = Dense(units=128, activation='relu')(bath_norm)
    
    # Reshape to prepare for Block 2
    reshaped_output = Reshape((-1, 128))(dense_layer_1)
    
    # Block 2
    for _ in range(3):  # Three branches for feature extraction
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
        
        # Branch 2: 1x7 convolution, 7x1 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: 3x3 convolution
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    
    # Concatenate outputs from branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])
    
    # Add batch normalization and dense layers after Block 2
    bath_norm = BatchNormalization()(concatenated_output)
    dense_layer_2 = Dense(units=64, activation='relu')(bath_norm)
    dense_layer_3 = Dense(units=10, activation='softmax')(dense_layer_2)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=dense_layer_3)
    
    return model