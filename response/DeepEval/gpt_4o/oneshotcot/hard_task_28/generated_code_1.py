import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Main path - add depthwise convolution layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add layer normalization
    layer_norm = LayerNormalization()(depthwise_conv)
    
    # Step 4: Add pointwise convolution layers
    pointwise_conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm)
    pointwise_conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pointwise_conv1)
    
    # Step 5: Branch path - directly connect to input
    branch_path = input_layer
    
    # Step 6: Combine main path and branch path using addition
    combined_output = Add()([pointwise_conv2, branch_path])
    
    # Step 7: Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Step 8: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Step 9: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model