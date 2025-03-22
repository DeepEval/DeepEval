import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input layer for CIFAR-10 dataset images
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define a block function that includes convolution, batch normalization, and ReLU activation
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # Create three parallel blocks
    block1_output = block(input_tensor=initial_conv)
    block2_output = block(input_tensor=initial_conv)
    block3_output = block(input_tensor=initial_conv)
    
    # Add the outputs of the blocks with the initial convolution's output
    added_output = Add()([initial_conv, block1_output, block2_output, block3_output])
    
    # Flatten the added output
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax activation for classification into 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model