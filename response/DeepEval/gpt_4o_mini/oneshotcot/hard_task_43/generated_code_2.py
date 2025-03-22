import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Conv2D, Reshape, BatchNormalization

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Block 1
    # Parallel paths with AveragePooling layers
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the outputs of the pooling layers
    flat_path1 = Flatten()(path1)
    flat_path2 = Flatten()(path2)
    flat_path3 = Flatten()(path3)
    
    # Concatenate the flattened outputs
    block1_output = Concatenate()([flat_path1, flat_path2, flat_path3])
    
    # Fully connected layer after Block 1
    dense_block1 = Dense(units=128, activation='relu')(block1_output)
    
    # Reshape layer to prepare for Block 2
    reshaped_output = Reshape((1, 1, 128))(dense_block1)
    
    # Step 3: Block 2
    # Branch 1: 1x1 and 3x3 convolutions
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshaped_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
    
    # Branch 2: 1x1, 1x7, 7x1 convolutions, and 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    
    # Branch 3: Average pooling
    branch3 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshaped_output)
    
    # Concatenate the outputs from all branches
    block2_output = Concatenate()([branch1, branch2, branch3])
    
    # Final fully connected layers for classification
    flatten_block2 = Flatten()(block2_output)
    dense_block2 = Dense(units=64, activation='relu')(flatten_block2)
    output_layer = Dense(units=10, activation='softmax')(dense_block2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model