import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        relu = Activation('relu')(input_tensor)
        separable_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(separable_conv)
        return max_pool

    block1 = block(input_layer)
    block2 = block(block1)

    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1)
    sum_path = Add()([block2, branch_path])

    bath_norm = BatchNormalization()(sum_path)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model