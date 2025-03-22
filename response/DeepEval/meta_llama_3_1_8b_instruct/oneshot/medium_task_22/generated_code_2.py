import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolution
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3x3)
    
    # Branch 2: 1x1 convolution followed by two 3x3 convolutions
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3_1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3x3_2)
    
    # Branch 3: max pooling
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    
    # Multi-scale feature fusion block
    def block(input_tensor1, input_tensor2, input_tensor3):
        output_tensor = Concatenate()([input_tensor1, input_tensor2, input_tensor3])
        output_tensor = BatchNormalization()(output_tensor)
        return output_tensor
        
    block_output = block(maxpool1, maxpool2, maxpool3)
    block_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output)
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model