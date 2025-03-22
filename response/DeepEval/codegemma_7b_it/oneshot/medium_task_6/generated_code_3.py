import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, MaxPooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):

        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = Activation('relu')(batch_norm)

        return relu
    
    block1_output = block(input_tensor=conv_initial)
    block2_output = block(input_tensor=block1_output)
    block3_output = block(input_tensor=block2_output)

    added_output = concatenate([conv_initial, block1_output, block2_output, block3_output])
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(added_output)

    flatten_layer = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model