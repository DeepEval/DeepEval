import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Add, Concatenate, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Depthwise convolution
        dw_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', name='depthwise_conv')(input_tensor)
        # Layer normalization
        ln = LayerNormalization(epsilon=1e-6)(dw_conv)
        # Pointwise convolution 1
        pwc1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='pointwise_conv1')(ln)
        # Pointwise convolution 2
        pwc2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='pointwise_conv2')(ln)
        
        return pwc2  # Output of the main path is the output of the second pointwise convolution
    
    main_output = main_path(input_tensor=input_layer)
    
    # Branch path
    def branch_path(input_tensor):
        # Directly outputs the input tensor as the branch output
        bp_output = input_tensor
        return bp_output
    
    branch_output = branch_path(input_tensor=input_layer)
    
    # Combine paths
    combined_output = Add()([main_output, branch_output])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()