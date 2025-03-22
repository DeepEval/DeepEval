import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    # Main Path
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)

    # Branch Path
    branch_path = input_layer  # Directly connected to input

    # Combine paths
    combined_output = Add()([main_path_conv2, branch_path])

    # Second Block
    def block_2(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)

        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor
    
    block2_output = block_2(input_tensor=combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model