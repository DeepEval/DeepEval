import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, LayerNormalization, Conv2D as PointwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Depthwise convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Layer normalization
        layer_norm = LayerNormalization()(depthwise_conv)
        # Pointwise convolution paths
        pointwise_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(layer_norm)
        pointwise_conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(pointwise_conv1)
        # Add branch path output
        add_branch = Add()([layer_norm, pointwise_conv1, pointwise_conv2])
        
        # Flatten and pass through two dense layers
        flatten = Flatten()(add_branch)
        dense1 = Dense(units=512, activation='relu')(flatten)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        
        return dense2

    # Branch path
    branch_output = main_path(input_tensor=input_layer)

    # Combine paths
    combined_output = Add()([branch_output, main_path(input_tensor=input_layer)])
    
    # Flatten and pass through two dense layers
    combined_flatten = Flatten()(combined_output)
    dense1 = Dense(units=512, activation='relu')(combined_flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model

# Build and return the model
model = dl_model()