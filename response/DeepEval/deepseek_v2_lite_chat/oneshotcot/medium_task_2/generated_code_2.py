import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        
        return pool
    
    main_output = main_path(input_tensor=input_layer)
    
    # Branch path
    def branch_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        flatten = Flatten()(conv)
        
        return flatten
    
    branch_output = branch_path(input_tensor=main_output)
    
    # Concatenate branch output with main output
    concat = Concatenate()([branch_output, main_output])
    
    # Batch normalization
    bn = BatchNormalization()(concat)
    
    # Flatten
    flat = Flatten()(bn)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model