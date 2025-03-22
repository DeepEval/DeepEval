import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def block(input_tensor):
    # Sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    
    # Parallel branch with a convolutional layer
    parallel_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
    
    # Addition operation
    output_tensor = Add()([conv3, parallel_conv])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First branch
    block1_output = block(input_layer)
    
    # Second branch
    block2_output = block(input_layer)
    
    # Concatenate the outputs of the two blocks
    concat_output = Concatenate()([block1_output, block2_output])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(concat_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build and return the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model