import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def conv_block(input_tensor, filters):
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    conv1_output = conv_block(input_tensor=input_layer, filters=32)
    conv2_output = conv_block(input_tensor=conv1_output, filters=64)
    conv3_output = conv_block(input_tensor=conv2_output, filters=128)
    
    concat_output = Concatenate(axis=-1)([conv1_output, conv2_output, conv3_output])
    batch_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model