import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Average pooling layers
    avg_pooling1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    avg_pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    avg_pooling3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Flatten and concatenate outputs
    flat_1 = Flatten()(avg_pooling1)
    flat_2 = Flatten()(avg_pooling2)
    flat_3 = Flatten()(avg_pooling3)
    concatenated = Concatenate()([flat_1, flat_2, flat_3])
    
    # Reshape concatenated output into a 4D tensor
    reshaped = Dense(4 * len(flat_1.shape), activation='relu')(concatenated)
    reshaped = keras.layers.Reshape((4, len(flat_1.shape)))(reshaped)
    
    # Second block
    def block(input_tensor):
        # Four parallel paths
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path2)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = keras.layers.Dropout(0.5)(path4)
        
        output_tensors = [path1, path2, path3, path4]
        path_outputs = [output for output in output_tensors if output.name != 'path1']
        path_concat = Concatenate()(path_outputs)
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(path_concat)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    block_output = block(reshaped)
    
    # Output model
    model = keras.Model(inputs=input_layer, outputs=block_output)
    
    return model