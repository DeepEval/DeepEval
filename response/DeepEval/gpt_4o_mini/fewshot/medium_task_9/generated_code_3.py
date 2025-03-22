import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, ReLU

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Initial convolutional layer to reduce dimensionality to 16
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    def basic_block(input_tensor):
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)

        # Branch path
        branch = input_tensor
        
        # Feature fusion via addition
        output_tensor = Add()([relu, branch])
        
        return output_tensor

    # First basic block
    block1_output = basic_block(initial_conv)

    # Second basic block
    block2_output = basic_block(block1_output)

    # Another convolutional layer in the branch to extract features
    branch_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)

    # Feature fusion again
    combined_output = Add()([block2_output, branch_conv])

    # Average pooling layer
    pooled_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(combined_output)

    # Flatten the output and add a fully connected layer
    flatten_layer = Flatten()(pooled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model