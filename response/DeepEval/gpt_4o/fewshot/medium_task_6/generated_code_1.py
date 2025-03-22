import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define a block with convolution, batch normalization, and ReLU activation
    def conv_block(input_tensor, filters, kernel_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        return relu

    # Three parallel blocks
    block1 = conv_block(input_tensor=initial_conv, filters=32, kernel_size=(3, 3))
    block2 = conv_block(input_tensor=initial_conv, filters=32, kernel_size=(5, 5))
    block3 = conv_block(input_tensor=initial_conv, filters=32, kernel_size=(7, 7))

    # Adding the outputs of the blocks with the initial convolution
    added_output = Add()([initial_conv, block1, block2, block3])

    # Flatten the added output
    flatten_layer = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model