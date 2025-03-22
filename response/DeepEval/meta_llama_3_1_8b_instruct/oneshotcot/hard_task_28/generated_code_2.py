import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    # Main path
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    layer_norm = LayerNormalization()(dw_conv)
    pw_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm)
    pw_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pw_conv1)
    
    # Branch path
    branch_path = input_layer
    
    # Combine the two paths
    combined_output = Add()([pw_conv2, branch_path])
    
    # Flatten and dense layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model