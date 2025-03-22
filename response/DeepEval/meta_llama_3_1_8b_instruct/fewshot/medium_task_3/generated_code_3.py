import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        
        return maxpool2
    
    block1_output = block_1(input_layer)

    # Define the second block
    def block_2(input_tensor):
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool3)
        maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        
        return maxpool4
    
    block2_output = block_2(input_layer)
    
    # Combine the outputs from both blocks
    adding_layer = Add()([block1_output, block2_output])
    
    # Add the input to the combined output
    input_adding_layer = Add()([adding_layer, input_layer])
    
    flatten_layer = Flatten()(input_adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model