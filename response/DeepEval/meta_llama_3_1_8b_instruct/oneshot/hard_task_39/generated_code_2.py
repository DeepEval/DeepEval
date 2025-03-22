import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Max Pooling with varying scales
    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor
    
    block_output_1 = block_1(input_layer)
    
    # Fully connected layer and reshape
    dense_1 = Dense(units=64, activation='relu')(block_output_1)
    reshape_layer = Reshape((1, 64))(dense_1)
    
    # Define the second block
    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Permute to match the output shape of the convolutional layers
        permute_layer = Permute((3, 1, 2))(maxpool)
        
        output_tensor = Concatenate()([conv1, conv2, conv3, permute_layer])
        return output_tensor
    
    block_output_2 = block_2(reshape_layer)
    
    # Batch normalization
    bath_norm = BatchNormalization()(block_output_2)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Classification output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model