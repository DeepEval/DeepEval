import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    # Parallel branch
    parallel_branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch1 = block(input_layer)

    # Branch 2
    branch2 = block(input_layer)

    # Concatenate all paths
    combined_output = keras.layers.add([branch1, branch2, parallel_branch])

    # Concatenate outputs from both blocks
    concatenated_output = keras.layers.concatenate([combined_output, combined_output], axis=-1)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model