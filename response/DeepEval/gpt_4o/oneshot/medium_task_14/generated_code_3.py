import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Define three sequential blocks
    def conv_block(x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # First block
    block1 = conv_block(input_layer, filters=32, kernel_size=(3, 3))
    
    # Second block
    block2 = conv_block(block1, filters=64, kernel_size=(3, 3))
    
    # Third block
    block3 = conv_block(block2, filters=128, kernel_size=(3, 3))

    # Parallel branch
    parallel_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Adding outputs from all paths
    added_output = Add()([block3, parallel_conv])

    # Fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model