import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Initial convolution layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    
    # Define a function for the parallel blocks
    def parallel_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm1 = BatchNormalization()(conv1)
        relu1 = keras.layers.Activation('relu')(norm1)
        
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm2 = BatchNormalization()(conv2)
        relu2 = keras.layers.Activation('relu')(norm2)

        conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm3 = BatchNormalization()(conv3)
        relu3 = keras.layers.Activation('relu')(norm3)

        return relu1, relu2, relu3

    # Create three parallel blocks
    block_outputs = parallel_block(initial_conv)
    
    # Add the outputs from the parallel blocks to the initial convolution's output
    combined_output = Add()([initial_conv, block_outputs[0], block_outputs[1], block_outputs[2]])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model