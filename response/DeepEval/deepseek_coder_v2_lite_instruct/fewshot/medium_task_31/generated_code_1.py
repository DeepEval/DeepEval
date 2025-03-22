import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def create_block(input_tensor):
        # Split the input tensor into three groups along the channel dimension
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply different convolutional kernels to each group
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layers[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_layers[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_layers[2])
        
        # Concatenate the outputs from the three groups
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        
        return concatenated

    # Create the main processing block
    block_output = create_block(input_tensor=input_layer)
    
    # Flatten the concatenated output
    flattened = Flatten()(block_output)
    
    # Pass the flattened output through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model