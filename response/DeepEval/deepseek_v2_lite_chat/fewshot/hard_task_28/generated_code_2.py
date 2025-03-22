import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    ln = LayerNormalization()(dw_conv)
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(ln)
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Branch path
    branch_output = Input(shape=(None, None, 64))  # Assuming 64 channels for the branch output
    
    # Addition operation
    combined_output = Add()([ln, branch_output])
    
    # Flattening and fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=[input_layer, branch_output], outputs=dense2)
    
    return model