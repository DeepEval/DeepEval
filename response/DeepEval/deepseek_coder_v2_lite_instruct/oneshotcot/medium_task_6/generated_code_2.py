import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Define the first block
    def block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Step 4: Apply the first block
    block1_output = block(conv)
    block2_output = block(conv)
    block3_output = block(conv)
    
    # Step 5: Add the outputs of the blocks to the initial convolution
    added_output = Add()([conv, block1_output, block2_output, block3_output])
    
    # Step 6: Flatten the result
    flatten_layer = Flatten()(added_output)
    
    # Step 7: Add dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 8: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model