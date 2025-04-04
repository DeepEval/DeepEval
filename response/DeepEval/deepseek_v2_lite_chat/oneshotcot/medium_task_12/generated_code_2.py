import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = ReLU()(block1_output)
    
    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(block1_output)
    block2_output = BatchNormalization()(block2_output)
    block2_output = ReLU()(block2_output)
    
    # Block 3
    block3_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(block2_output)
    block3_output = BatchNormalization()(block3_output)
    block3_output = ReLU()(block3_output)
    
    # Concatenate outputs of each block along the channel dimension
    concatenated_output = Concatenate(axis=-1)([block1_output, block2_output, block3_output])
    
    # Flatten and pass through two fully connected layers
    flattened_output = Flatten()(concatenated_output)
    dense1 = Dense(units=256, activation='relu')(flattened_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model