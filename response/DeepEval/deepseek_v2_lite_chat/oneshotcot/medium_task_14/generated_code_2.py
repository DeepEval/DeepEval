import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = ReLU()(block1_output)
    
    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    block2_output = BatchNormalization()(block2_output)
    block2_output = ReLU()(block2_output)
    
    # Block 3
    block3_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    block3_output = BatchNormalization()(block3_output)
    block3_output = ReLU()(block3_output)
    
    # Parallel branch for direct input processing
    parallel_branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    parallel_branch_output = BatchNormalization()(parallel_branch_output)
    parallel_branch_output = ReLU()(parallel_branch_output)
    
    # Concatenate outputs from sequential blocks and parallel branch
    concatenated_output = Concatenate()([block1_output, block2_output, block3_output, parallel_branch_output])
    
    # Flatten and add dense layers
    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model