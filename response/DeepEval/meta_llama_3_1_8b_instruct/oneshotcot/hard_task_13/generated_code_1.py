import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the first block with four parallel branches
    def block1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Define the second block with global average pooling and two fully connected layers
    block2_output = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Reshape the weights of the second fully connected layer to match the shape of the first block's output
    weights = Dense(units=32*32*32, activation='linear')(dense2)
    weights = Reshape((32, 32, 32))(weights)
    
    # Multiply the reshaped weights with the first block's output
    multiplied_output = Multiply()([block1_output, weights])
    
    # Define the final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(multiplied_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model