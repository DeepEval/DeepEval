from keras.models import Model
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Conv2D, AveragePooling2D, Dense, Reshape 


def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    def block1(input_tensor):
        
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        drop1 = Dropout(rate=0.2)(flatten1)

        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        drop2 = Dropout(rate=0.2)(flatten2)

        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        drop3 = Dropout(rate=0.2)(flatten3)

        output_tensor = Concatenate()([drop1, drop2, drop3])

        return output_tensor

    def block2(input_tensor):

        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_tensor)

        conv_2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_2_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_2_1)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(2, 2), padding='same', activation='relu')(conv_2_2)

        conv_3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_3_2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_3_1)
        conv_3_3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_3_2)
        conv_3_4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_3_3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(2, 2), padding='same', activation='relu')(conv_3_4)

        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)

        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block_output1 = block1(input_tensor=input_layer)
    dense_block1 = Dense(units=256, activation='relu')(block_output1)
    reshaped = Reshape(target_shape=(16, 16, 1))(dense_block1)
    block_output2 = block2(input_tensor=reshaped)

    dense1 = Dense(units=128, activation='relu')(block_output2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model