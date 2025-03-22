import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        return conv3

    # First branch
    branch1_output = block(input_layer)
    
    # Second branch with an additional convolution layer
    branch2_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_output = block(branch2_input)
    
    # Combine outputs from both branches using addition
    combined_output = Add()([branch1_output, branch2_output])
    
    # Concatenate the outputs from both blocks
    concat_output = Concatenate()([branch1_output, branch2_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_output)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model