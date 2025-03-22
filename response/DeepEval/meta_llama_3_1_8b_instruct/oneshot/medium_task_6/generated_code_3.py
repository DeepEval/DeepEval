import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        relu1 = ReLU()(bn1)
        
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn2 = BatchNormalization()(conv2)
        relu2 = ReLU()(bn2)
        
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn3 = BatchNormalization()(conv3)
        relu3 = ReLU()(bn3)

        output_tensor = Add()([relu1, relu2, relu3])

        return output_tensor
        
    block_output1 = block(conv)
    block_output2 = block(block_output1)
    block_output3 = block(block_output2)
    added_output = Add()([conv, block_output1, block_output2, block_output3])

    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model