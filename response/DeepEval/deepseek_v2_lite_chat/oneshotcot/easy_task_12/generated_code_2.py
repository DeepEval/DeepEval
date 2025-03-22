import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def block(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return pool2
    
    main_path_output = block(input_tensor=input_layer)
    
    # Branch path
    branch_path_input = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(main_path_output)
    
    # Sum main and branch paths
    merged = Concatenate()([main_path_output, branch_path_input])
    
    # Flatten and Fully Connected Layers
    flatten = Flatten()(merged)
    dense = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense)
    output = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model