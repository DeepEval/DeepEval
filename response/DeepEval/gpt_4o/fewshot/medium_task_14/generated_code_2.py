import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters, kernel_size=(3, 3)):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        activated = ReLU()(norm)
        return activated

    # Block 1
    block1_output = block(input_layer, filters=32)
    
    # Block 2
    block2_output = block(block1_output, filters=64)
    
    # Block 3
    block3_output = block(block2_output, filters=128)

    # Parallel branch processing the input directly
    parallel_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine all paths
    combined_output = Add()([block1_output, block2_output, block3_output, parallel_conv])

    # Fully connected layers
    flatten = Flatten()(combined_output)
    fc1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model