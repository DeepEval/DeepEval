import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(input_layer)
    layer_norm = LayerNormalization()(depthwise_conv)
    pointwise_conv1 = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1), activation='relu')(layer_norm)
    pointwise_conv2 = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1), activation='relu')(pointwise_conv1)
    
    # Branch Path
    branch_output = input_layer
    
    # Addition of Main Path and Branch Path
    added_output = Add()([pointwise_conv2, branch_output])
    
    # Flattening the output
    flattened_output = Flatten()(added_output)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model