import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    # Main Path
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    
    # Branch Path (direct connection)
    branch = input_layer
    
    # Combining the main path and branch path
    block1_output = Add()([conv2_main, branch])

    # Second Block
    # Three MaxPooling paths with varying scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)
    
    # Flatten and Concatenate the outputs of the pooling layers
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)
    
    block2_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model