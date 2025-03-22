import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: three average pooling layers with pooling windows and strides of 1x1, 2x2, and 4x4
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)
    block1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1)
    block1 = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1)
    block1 = Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1)
    
    # Second block: flatten and concatenate the outputs of the three pooling layers
    flattened_block1 = Flatten()(block1)
    block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(flattened_block1)
    block2 = Flatten()(block2)
    block3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(flattened_block1)
    block3 = Flatten()(block3)
    block4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(flattened_block1)
    block4 = Flatten()(block4)
    fused_block = keras.layers.concatenate([block2, block3, block4], axis=1)
    
    # Third block: fully connected layers
    dense1 = Dense(units=64, activation='relu')(fused_block)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model