import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    drop_out_1 = Dropout(0.2)(conv)

    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(drop_out_1)

    def block(input_tensor):

        path1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
        
    block_output = block(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_output)
    drop_out_2 = Dropout(0.2)(bath_norm)
    flatten_layer = Flatten()(drop_out_2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model