import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolution
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Three Parallel Blocks
    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        relu1 = ReLU()(bn1)
        
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu1)
        bn2 = BatchNormalization()(conv2)
        relu2 = ReLU()(bn2)
        
        conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu2)
        bn3 = BatchNormalization()(conv3)
        relu3 = ReLU()(bn3)
        
        # Add the outputs of the blocks
        added_output = Add()([conv, relu3])
        return added_output
    
    block1_output = block(input_layer)
    block2_output = block(block1_output)
    block3_output = block(block2_output)
    
    # Flatten the output and pass through fully connected layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model