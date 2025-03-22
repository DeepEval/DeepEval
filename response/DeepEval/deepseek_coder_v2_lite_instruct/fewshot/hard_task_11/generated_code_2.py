import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch paths
    conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1x3_branch = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x1_branch = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge branch outputs
    merged_branch = Concatenate(axis=-1)([conv1x1_branch, conv1x3_branch, conv3x1_branch])
    
    # Add main pathway and merged branch
    added_output = Add()([conv1x1_main, merged_branch])
    
    # Another 1x1 convolution
    final_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(added_output)
    
    # Direct connection from input to branch
    direct_connection = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add direct connection and final convolution
    final_output = Add()([final_conv, direct_connection])
    
    # Flatten the final output
    flattened_output = Flatten()(final_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model