import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # First block
    batch_norm1 = BatchNormalization()(conv1)
    block1 = ReLU()(batch_norm1)
    
    # Second block
    batch_norm2 = BatchNormalization()(block1)
    block2 = ReLU()(batch_norm2)
    
    # Third block
    batch_norm3 = BatchNormalization()(block2)
    block3 = ReLU()(batch_norm3)
    
    # Concatenate the outputs of the blocks with the initial convolution
    concat_layer = Concatenate(axis=-1)([conv1, block1, block2, block3])
    
    # Flatten and pass through dense layers
    flatten = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model