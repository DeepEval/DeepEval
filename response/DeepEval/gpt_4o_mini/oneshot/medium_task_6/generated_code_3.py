import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Initial Convolution
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    initial_bn = BatchNormalization()(initial_conv)
    initial_relu = ReLU()(initial_bn)

    # Define the three parallel blocks
    def parallel_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)
        return relu
    
    block1_output = parallel_block(initial_relu)
    block2_output = parallel_block(initial_relu)
    block3_output = parallel_block(initial_relu)

    # Adding the outputs of the blocks to the initial convolution's output
    added_output = Add()([initial_relu, block1_output, block2_output, block3_output])

    # Flattening the output
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model