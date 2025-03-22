import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D, BatchNormalization

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Paths with different average pooling sizes
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten and Dropout
    flatten1 = Flatten()(path1)
    flatten2 = Flatten()(path2)
    flatten3 = Flatten()(path3)
    
    dropout1 = Dropout(0.5)(flatten1)
    dropout2 = Dropout(0.5)(flatten2)
    dropout3 = Dropout(0.5)(flatten3)
    
    # Concatenate the flattened outputs
    block1_output = Concatenate()([dropout1, dropout2, dropout3])

    # Fully connected layer and reshape
    fc1 = Dense(128, activation='relu')(block1_output)
    reshaped_output = Reshape((2, 2, 32))(fc1)  # Reshape output to a suitable format for Block 2

    # Block 2
    # Branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    
    branch4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshaped_output)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

    # Concatenate branch outputs
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the output of Block 2
    flattened_block2_output = Flatten()(block2_output)
    
    # Fully connected layers for final output
    final_fc1 = Dense(128, activation='relu')(flattened_block2_output)
    final_output = Dense(10, activation='softmax')(final_fc1)  # Assuming 10 classes for MNIST

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model