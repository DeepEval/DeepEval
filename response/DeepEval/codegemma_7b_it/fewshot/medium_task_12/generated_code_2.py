import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(bn1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(act1)
        bn2 = BatchNormalization()(conv2)
        act2 = Activation('relu')(bn2)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2)
        bn3 = BatchNormalization()(conv3)
        act3 = Activation('relu')(bn3)
        concat = concatenate([act1, act2, act3], axis=-1)
        return concat
    
    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=block1_output)
    block3_output = block(input_tensor=block2_output)
    flatten = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model