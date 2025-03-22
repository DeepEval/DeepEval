import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Main Path and Branch Path with Addition
    # Main path
    conv_main_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main_2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_1)
    # Branch path
    branch_path = input_layer
    # Combining paths
    block1_output = Add()([conv_main_2, branch_path])

    # Block 2: Multiple Max Pooling Layers with Concatenation
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    flatten1 = Flatten()(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    flatten2 = Flatten()(pool2)
    
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)
    flatten3 = Flatten()(pool3)
    
    block2_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully Connected Layers for Classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model