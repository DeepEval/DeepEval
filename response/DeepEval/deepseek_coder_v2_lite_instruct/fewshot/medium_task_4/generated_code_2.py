import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution followed by average pooling
    def path1_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool
    
    # Apply path1_block twice
    path1_output = path1_block(input_layer)
    path1_output = path1_block(path1_output)
    
    # Path 2: Single convolutional layer
    path2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_output)
    path2_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path2_output)
    
    # Combine outputs from both paths using addition
    combined_output = Add()([path1_output, path2_output])
    
    # Flatten the combined output
    flattened_output = Flatten()(combined_output)
    
    # Fully connected layer to produce the final classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model