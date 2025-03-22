import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.3)(depthwise_conv)
        pointwise_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.3)(pointwise_conv)
        return dropout2
    
    # Create three branches
    branch1_output = specialized_block(input_tensor=input_layer)
    branch2_output = specialized_block(input_tensor=input_layer)
    branch3_output = specialized_block(input_tensor=input_layer)
    
    # Concatenate the outputs of the three branches
    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    
    # Fully connected layers for classification
    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model