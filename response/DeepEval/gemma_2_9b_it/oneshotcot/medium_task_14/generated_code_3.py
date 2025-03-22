import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():  

    input_layer = Input(shape=(32, 32, 3)) 

    # Parallel Branch
    x_parallel = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    x_block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_block1 = BatchNormalization()(x_block1)
    
    # Block 2
    x_block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_block2 = BatchNormalization()(x_block2)

    # Block 3
    x_block3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_block3 = BatchNormalization()(x_block3)
    
    # Concatenate all paths
    merged = Concatenate()([x_parallel, x_block1, x_block2, x_block3]) 

    # Flatten and Dense Layers
    flattened = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model