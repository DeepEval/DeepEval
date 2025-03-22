import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Define the block
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        output_tensor1 = path1
        output_tensor2 = path2
        output_tensor3 = path3
        return output_tensor1, output_tensor2, output_tensor3
    
    # Create the first block
    output_tensor11, output_tensor12, output_tensor13 = block(input_layer)
    output_tensor1 = Add()([output_tensor11, output_tensor12, output_tensor13])
    
    # Create the second block
    output_tensor21, output_tensor22, output_tensor23 = block(input_layer)
    output_tensor2 = Add()([output_tensor21, output_tensor22, output_tensor23])
    
    # Concatenate the outputs from both blocks
    concat_output = Concatenate()([output_tensor1, output_tensor2])
    
    # Add a parallel branch
    parallel_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat_output = Add()([concat_output, parallel_output])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Add fully connected layers to produce the final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model