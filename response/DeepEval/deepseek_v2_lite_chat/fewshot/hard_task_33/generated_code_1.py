import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)
    
    # Define the input tensor
    input_layer = Input(shape=input_shape)
    
    # Define the first branch
    def branch_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        return Add()([conv1, conv2])
    
    # Define the second branch
    def branch_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        return Add()([depthwise_conv1, conv2])
    
    # Define the third branch
    def branch_3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        return Add()([depthwise_conv1, conv2])
    
    # Apply each branch to the input
    branch1_output = branch_1(input_layer)
    branch2_output = branch_2(input_layer)
    branch3_output = branch_3(input_layer)
    
    # Concatenate the outputs from all branches
    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    
    # Flatten the concatenated output and pass it through a fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model