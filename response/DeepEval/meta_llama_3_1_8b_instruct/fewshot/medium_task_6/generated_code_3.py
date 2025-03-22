import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, Lambda, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolution
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Blocks
    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        output_tensor = conv1 + batch_norm1
        return output_tensor
    
    block1 = block(conv)
    block2 = block(block1)
    block3 = block(block2)
    
    # Add Outputs
    adding_layer = Add()([conv, block1, block2, block3])
    
    # Flatten and Classification
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model