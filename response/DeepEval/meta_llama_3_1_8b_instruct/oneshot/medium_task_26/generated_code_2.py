import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))
    
    # Compress the input channels with a 1x1 convolutional layer
    conv_compress = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Expand the features through two parallel convolutional layers
    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2])

        return output_tensor
        
    block_output = block(conv_compress)
    
    # Apply batch normalization to the concatenated output
    bath_norm = BatchNormalization()(block_output)
    
    # Flatten the feature map into a one-dimensional vector
    flatten_layer = Flatten()(bath_norm)
    
    # Pass the flattened vector through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model