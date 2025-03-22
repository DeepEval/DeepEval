import keras
from keras.layers import Input, Conv2D, Add, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Define the block
    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Add the paths and apply dropout
        output_tensor = Add()([path1, path2, path3, path4, path5])
        output_tensor = Dropout(0.2)(output_tensor)

        return output_tensor
        
    # Apply the block to the input
    block_output = block(input_layer)
    
    # Add the original input to the block output
    add_output = Add()([input_layer, block_output])
    
    # Apply another block to the added output
    block_output_2 = block(add_output)
    
    # Apply dropout to the block output
    block_output_2 = Dropout(0.2)(block_output_2)
    
    # Apply a 1x1 convolutional layer to restore the number of channels
    conv_output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_output_2)
    
    # Flatten the output
    flatten_layer = Flatten()(conv_output)
    
    # Apply a fully connected layer to produce the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model