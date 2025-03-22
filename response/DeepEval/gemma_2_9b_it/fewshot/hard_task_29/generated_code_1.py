import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch path
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    output_block1 = Add()([conv2, branch])

    # Block 2
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(output_block1)
    flatten1 = Flatten()(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output_block1)
    flatten2 = Flatten()(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(output_block1)
    flatten3 = Flatten()(maxpool3)
    output_block2 = Concatenate()([flatten1, flatten2, flatten3])

    # Output Layers
    dense1 = Dense(units=64, activation='relu')(output_block2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model