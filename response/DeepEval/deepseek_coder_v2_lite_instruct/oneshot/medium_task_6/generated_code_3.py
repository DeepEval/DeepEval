import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, ReLU
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def block(input_tensor):
        # Convolutional layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Batch normalization
        batch_norm = BatchNormalization()(conv)
        # ReLU activation
        relu = ReLU()(batch_norm)
        return relu
    
    # First block
    block1_output = block(conv_initial)
    # Second block
    block2_output = block(block1_output)
    # Third block
    block3_output = block(block2_output)
    
    # Add the initial convolution's output to the block outputs
    added_output = Add()([conv_initial, block1_output, block2_output, block3_output])
    
    # Flatten the output
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model