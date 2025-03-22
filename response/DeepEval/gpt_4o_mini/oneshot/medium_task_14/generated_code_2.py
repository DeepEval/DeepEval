import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Define three sequential blocks
    def conv_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        return relu

    # Creating three separate blocks
    block1_output = conv_block(input_layer)
    block2_output = conv_block(block1_output)
    block3_output = conv_block(block2_output)
    
    # Parallel branch processing input directly with convolution
    parallel_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    
    # Adding the outputs from all paths
    added_output = Add()([block3_output, parallel_conv])
    
    # Flattening the output for the dense layers
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model