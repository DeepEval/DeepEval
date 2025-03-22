import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, UpSampling2D, Dropout

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        skip_conn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Add()([conv2, skip_conn])

        return output_tensor
        
    block_output1 = block(max_pooling2)
    up_sample1 = UpSampling2D(size=(2, 2))(block_output1)
    concat1 = Concatenate()([conv2, up_sample1])

    block_output2 = block(concat1)
    up_sample2 = UpSampling2D(size=(2, 2))(block_output2)
    concat2 = Concatenate()([max_pooling2, up_sample2])

    dropout_layer = Dropout(0.2)(concat2)
    conv_final = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_layer)
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv_final)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model