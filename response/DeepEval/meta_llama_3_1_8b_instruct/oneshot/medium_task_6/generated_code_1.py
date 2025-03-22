import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        batch_norm1 = BatchNormalization()(conv1)
        batch_norm2 = BatchNormalization()(conv2)
        batch_norm3 = BatchNormalization()(conv3)
        
        output_tensor = Add()([batch_norm1, batch_norm2, batch_norm3])

        return output_tensor
        
    block1_output = block(conv)
    block2_output = block(block1_output)
    block3_output = block(block2_output)
    added_output = Add()([conv, block1_output, block2_output, block3_output])

    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model