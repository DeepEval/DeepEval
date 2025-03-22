import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

        return pool1
    
    first_block_output = first_block(input_tensor=input_layer)
    
    # Second Block
    def second_block(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        return dense2
    
    second_block_output = second_block(input_tensor=first_block_output)
    
    # Concatenate the outputs
    concatenated_layer = Concatenate()([first_block_output, second_block_output])
    
    # Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model