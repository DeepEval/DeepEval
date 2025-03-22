import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Add, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block_output_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block_output_1 = BatchNormalization()(block_output_1)
    
    # Block 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output_1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    block_output_2 = BatchNormalization()(conv2)
    
    # Block 3
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output_2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    block_output_3 = BatchNormalization()(conv3)
    
    # Concatenate the outputs of each block
    output_tensor = Concatenate()([block_output_1, block_output_2, block_output_3])
    
    # Flatten the output
    flatten_layer = Flatten()(output_tensor)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model