import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    def block(input_tensor):
        # Convolutional layer with Batch Normalization and ReLU activation
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        
        return relu

    # First block
    block1_output = block(input_layer)

    # Second block
    block2_output = block(block1_output)
    
    # Concatenate the first and second blocks
    concatenated_block = Concatenate()([block1_output, block2_output])

    # Third block
    block3_output = block(concatenated_block)
    
    # Concatenate the previous blocks
    concatenated_output = Concatenate()([concatenated_block, block3_output])

    # Flatten the output from the last block
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model