import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def block(input_tensor):
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # Third convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    return conv3

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First parallel branch
    block_output1 = block(input_layer)
    
    # Second parallel branch (same as the first)
    block_output2 = block(input_layer)
    
    # Additional branch (convolutional layer connected to input)
    additional_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from the branches using addition
    combined_output1 = Add()([block_output1, block_output2])
    combined_output2 = Add()([combined_output1, additional_conv])
    
    # Concatenate outputs from the two blocks
    concatenated_output = Concatenate()([combined_output1, combined_output2])
    
    # Flatten the result
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer for classification (10 classes for MNIST)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model