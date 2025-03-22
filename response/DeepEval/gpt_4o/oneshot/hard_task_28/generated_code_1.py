import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Add, Flatten, Dense, LayerNormalization
from keras.models import Model

def dl_model():
    # Input layer with CIFAR-10 image shape
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    layer_norm = LayerNormalization()(depthwise_conv)
    
    # First pointwise convolution
    pointwise_conv1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(layer_norm)
    # Second pointwise convolution
    pointwise_conv2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(pointwise_conv1)

    # Branch path (direct connection to input)
    branch_path = input_layer

    # Combine the main path and branch path using addition
    combined_output = Add()([pointwise_conv2, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model