import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Paths for Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    # Concatenate paths from Block 1
    block1_output = Concatenate()([path1, path2, path3])
    
    # Flatten and dropout
    flattened_block1 = Flatten()(block1_output)
    dropout_block1 = Dropout(0.5)(flattened_block1)
    
    # Fully connected layer to reshape the output
    reshape_layer = Dense(16, activation='relu')(dropout_block1)
    
    # Block 2
    # Paths for Block 2
    path1_block2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    
    path2_block2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path2_block2 = Conv2D(64, (1, 7), activation='relu')(path2_block2)
    path2_block2 = Conv2D(64, (7, 1), activation='relu')(path2_block2)
    
    path3_block2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path3_block2 = Conv2D(64, (7, 1), activation='relu')(path3_block2)
    path3_block2 = Conv2D(64, (1, 7), activation='relu')(path3_block2)
    
    path4_block2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path4_block2 = AveragePooling2D((4, 4))(path4_block2)
    path4_block2 = Conv2D(64, (1, 1), activation='relu')(path4_block2)
    
    # Concatenate paths from Block 2
    block2_output = Concatenate(axis=3)([path1_block2, path2_block2, path3_block2, path4_block2])
    
    # Flatten and dropout
    flattened_block2 = Flatten()(block2_output)
    dropout_block2 = Dropout(0.5)(flattened_block2)
    
    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(dropout_block2)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model