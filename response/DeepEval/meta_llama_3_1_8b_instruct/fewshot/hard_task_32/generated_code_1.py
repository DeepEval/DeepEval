import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def branch_block(input_tensor):
        
        conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        conv3 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv3 = Dropout(0.2)(conv3)
        
        output_tensor = Concatenate()([conv1, conv2, conv3])
        
        return output_tensor
    
    branch1_output = branch_block(input_layer)
    branch2_output = branch_block(branch1_output)
    branch3_output = branch_block(branch2_output)
    
    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output])
    flatten = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model