import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First branch
    def branch_1(input_tensor):
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_out = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv1)
        
        return conv1_out
    
    # Second branch
    def branch_2(input_tensor):
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        conv2_out = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv2)
        
        return conv2_out
    
    # Third branch
    def branch_3(input_tensor):
        conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
        conv3_out = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3)
        
        return conv3_out
    
    # Process each branch
    conv1_output = branch_1(input_tensor=input_layer)
    conv2_output = branch_2(input_tensor=input_layer)
    conv3_output = branch_3(input_tensor=input_layer)
    
    # Concatenate the outputs
    concat_layer = Concatenate(axis=-1)([conv1_output, conv2_output, conv3_output])
    
    # Flatten and fully connected layers
    flatten = Flatten()(concat_layer)
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()