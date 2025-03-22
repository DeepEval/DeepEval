import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    conv_main = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel branch
    path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Concatenate()([path1, path2, path3])
    
    # Concatenate the two paths
    output_tensor = Concatenate()([conv_main, path4])
    
    # Add 1x1 convolution to the concatenated output
    output_tensor = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    
    # Add direct connection from input to the branch
    direct_connection = input_layer
    output_tensor = Add()([output_tensor, direct_connection])
    
    # Batch normalization
    bath_norm = BatchNormalization()(output_tensor)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model