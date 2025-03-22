import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First feature extraction path
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second feature extraction path
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Concatenate the outputs of the two paths
    concat = Concatenate()([path1, path2])
    
    # Apply a 1x1 convolution to align the output dimensions
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Branch connecting directly to the input
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge the outputs of the main path and the branch through addition
    merged = keras.layers.add([main_path, branch])
    
    # Batch normalization
    batch_norm = BatchNormalization()(merged)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model