import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Define the block
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv2)
        return avg_pool

    # Apply the block twice
    block_output = block(input_tensor)
    block_output = block(block_output)

    # Add dropout and global average pooling
    dropout = Dropout(rate=0.2)(block_output)
    global_avg_pool = GlobalAveragePooling2D()(dropout)

    # Flatten and fully connected layers
    flatten = Flatten()(global_avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model