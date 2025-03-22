import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the first path with a 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define the second path with a sequence of convolutions
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_1x7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
    conv_7x1 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x7)
    path2 = Concatenate()([conv_1x1, conv_1x7, conv_7x1])
    
    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])
    
    # Apply a 1x1 convolution to align the output dimensions
    conv_align = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Create a branch that connects directly to the input
    branch = input_layer
    
    # Merge the outputs of the main path and the branch through addition
    merged = Add()([conv_align, branch])
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(merged)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Define the second fully connected layer
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model