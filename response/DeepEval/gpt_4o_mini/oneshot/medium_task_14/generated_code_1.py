import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def conv_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu
    
    # Creating three convolutional blocks
    block1_output = conv_block(input_layer)
    block2_output = conv_block(block1_output)
    block3_output = conv_block(block2_output)

    # Parallel branch of convolutional layers processing the input directly
    parallel_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_batch_norm = BatchNormalization()(parallel_conv)
    parallel_relu = ReLU()(parallel_batch_norm)

    # Adding outputs from all paths
    merged_output = Add()([block3_output, parallel_relu])

    # Flatten the result
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model