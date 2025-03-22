import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    def block(input_tensor):
        # Main sequential convolutional path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        # Parallel path
        parallel_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Combine outputs using addition
        added_output = Add()([conv1, conv2, conv3, parallel_conv])
        
        return added_output
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First branch
    branch1_output = block(input_layer)
    
    # Second branch
    branch2_output = block(input_layer)
    
    # Concatenate outputs of the two branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])
    
    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model