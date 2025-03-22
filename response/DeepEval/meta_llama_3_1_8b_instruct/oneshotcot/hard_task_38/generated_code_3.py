import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # define the block
    def block(input_tensor):
        
        # first path: batch normalization and ReLU activation
        batch_norm = BatchNormalization()(input_tensor)
        relu = Activation('relu')(batch_norm)
        
        # second path: 3x3 convolutional layer
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        
        # merge the original input with the new features along the channel dimension
        output_tensor = Concatenate()([input_tensor, conv])
        
        return output_tensor
    
    # define the pathway
    pathway1 = block(input_layer)
    pathway2 = block(pathway1)
    pathway3 = block(pathway2)
    
    # merge the outputs from both pathways through concatenation
    merged_output = Concatenate()([pathway1, pathway2, pathway3])
    
    # flatten the merged output
    flatten_layer = Flatten()(merged_output)
    
    # two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model