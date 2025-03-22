import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel paths in the main path
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Concatenate the outputs of the parallel paths
    concatenated_main = keras.layers.concatenate([path1, path2])
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from the main and branch paths
    added_output = Add()([concatenated_main, branch_path])
    
    # Batch normalization
    batch_norm = BatchNormalization()(added_output)
    
    # Flatten the output
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model