import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def create_block(input_tensor):
    # Define three sequential convolutional layers in the block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    return conv3

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First branch
    branch1 = create_block(input_layer)
    
    # Second branch
    branch2 = create_block(input_layer)
    
    # Parallel branch connected through a convolutional layer
    parallel_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding outputs from both branches and the parallel branch
    combined = Add()([branch1, branch2, parallel_branch])
    
    # Concatenate the two blocks' outputs
    concatenated = Concatenate()([branch1, branch2])
    
    # Flattening the result
    flatten_layer = Flatten()(combined)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model