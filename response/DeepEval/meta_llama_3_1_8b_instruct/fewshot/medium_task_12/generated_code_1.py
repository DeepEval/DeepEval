import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 3 color channels
    
    def block(input_tensor, filters):
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        max_pool = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(bn2)
        return max_pool

    block1_output = block(input_layer, 32)
    block2_output = block(block1_output, 64)
    block3_output = block(block2_output, 128)
    
    # Concatenate outputs from each block
    output_tensor = Concatenate()([input_layer, block1_output, block2_output, block3_output])
    
    flatten_layer = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model