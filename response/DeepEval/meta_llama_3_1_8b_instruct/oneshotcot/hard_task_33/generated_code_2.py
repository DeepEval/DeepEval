import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Define the block
    def block(input_tensor):

        # Elevate the dimension through a 1x1 convolutional layer
        elevate = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Extract features through a 3x3 depthwise separable convolutional
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(elevate)
        depthwise = BatchNormalization()(depthwise)
        depthwise ='relu' if keras.backend.image_data_format() == 'channels_last' else'relu6'
        depthwise = keras.layers.Activation(depthwise)(depthwise)
        
        # Reduce the dimension through a 1x1 convolutional layer
        reduce = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
        
        # Add the block's input to form the output
        output_tensor = Add()([input_tensor, reduce])
        
        return output_tensor
    
    # Three branches
    branch1 = block(input_layer)
    branch2 = block(branch1)
    branch3 = block(branch2)
    
    # Concatenate the outputs from the three branches
    output_tensor = Concatenate()([branch1, branch2, branch3])
    
    # Flattening layer
    flatten_layer = Flatten()(output_tensor)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model