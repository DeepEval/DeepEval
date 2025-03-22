import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_stage1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_stage1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_stage1)

    conv_stage2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_stage1)
    max_pooling_stage2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_stage2)

    def down_block(input_tensor):

        conv_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_path)
        max_pooling_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)

        output_tensor = Concatenate()([conv_path, max_pooling_path])

        return output_tensor
        
    down_block_output = down_block(max_pooling_stage2)
    conv_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(down_block_output)
    conv_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_path)
    dropout_path = Dropout(0.2)(conv_path)

    def up_block(input_tensor, skip_connection):

        up_sampling = UpSampling2D(size=(2, 2))(input_tensor)
        conv_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_sampling)
        conv_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_path)
        add_path = Add()([conv_path, skip_connection])

        return add_path
        
    up_block_output = up_block(dropout_path, max_pooling_stage2)
    up_block_output = up_block(up_block_output, conv_stage2)

    conv_last = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(up_block_output)

    model = keras.Model(inputs=input_layer, outputs=conv_last)

    return model