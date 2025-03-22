import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers import Add
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Define initial convolutional layer
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define basic block
    def basic_block(input_tensor):
        
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Add()([main_path, branch])
        
        return output_tensor
    
    # Apply basic block twice with an intermediate convolutional layer
    block1 = basic_block(initial_conv)
    intermediate_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = basic_block(intermediate_conv)
    
    # Add outputs from both paths and apply average pooling
    output_tensor = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(Add()([block1, block2]))
    
    # Flatten the output and add a fully connected layer
    flatten_layer = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model