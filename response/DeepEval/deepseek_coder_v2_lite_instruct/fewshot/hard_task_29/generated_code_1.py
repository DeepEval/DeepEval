import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    def first_block(input_tensor):
        # Main Path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        # Branch Path
        branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Addition of Main Path and Branch Path
        added = Add()([conv2, branch])
        return added

    block1_output = first_block(input_layer)

    # Second Block
    def second_block(input_tensor):
        # Max Pooling Layers
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        # Flattening and Concatenation
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        concatenated = Dense(units=128, activation='relu')(Concatenate()([flatten1, flatten2, flatten3]))
        return concatenated

    block2_output = second_block(block1_output)

    # Fully Connected Layers
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model