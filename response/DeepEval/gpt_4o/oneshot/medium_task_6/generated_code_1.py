import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define the block
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        return relu

    # Create three parallel blocks
    block1 = block(input_tensor=initial_conv)
    block2 = block(input_tensor=initial_conv)
    block3 = block(input_tensor=initial_conv)

    # Add the outputs of the blocks and the initial convolution
    added_output = Add()([initial_conv, block1, block2, block3])
    
    # Flatten, fully connected layers, and output
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model