import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Activation, SeparableConv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    block_output = block(input_tensor=max_pooling1)
    
    for i in range(2):
        block_output = block(input_tensor=block_output)
        
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    branch_conv = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv)
    
    # Fuse paths
    fused_output = keras.layers.add([branch_conv, block_output])
    
    # Output layer
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def block(input_tensor):
    
    # Block specific operations
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Concatenate operations
    output_tensor = Concatenate(axis=3)([conv1, conv2, conv3])
    
    return output_tensor