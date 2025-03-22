import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Add, Concatenate

def block(input_tensor):
    # Three sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    return conv3

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First parallel branch using the defined block
    block_output1 = block(input_layer)

    # Second parallel branch using the same block
    block_output2 = block(input_layer)

    # Parallel branch that connects the input through a convolutional layer
    conv_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combining outputs using an addition operation
    combined_output = Add()([block_output1, block_output2, conv_branch])

    # Flattening the output for the dense layer
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model