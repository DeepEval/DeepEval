import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):
        maxpool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        maxpool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        maxpool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)

        maxpool_1x1_flatten = Flatten()(maxpool_1x1)
        maxpool_2x2_flatten = Flatten()(maxpool_2x2)
        maxpool_4x4_flatten = Flatten()(maxpool_4x4)

        maxpool_1x1_dropout = Dropout(0.2)(maxpool_1x1_flatten)
        maxpool_2x2_dropout = Dropout(0.2)(maxpool_2x2_flatten)
        maxpool_4x4_dropout = Dropout(0.2)(maxpool_4x4_flatten)

        output_tensor = Concatenate()([maxpool_1x1_dropout, maxpool_2x2_dropout, maxpool_4x4_dropout])

        return output_tensor

    block1_output = block1(input_tensor=input_layer)
    block1_reshaped = Reshape((block1_output.shape[1] * block1_output.shape[2],))(block1_output)

    block2_input = Dense(units=256, activation='relu')(block1_reshaped)

    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        path_concat = Concatenate()([path1, path2, path3, path4])

        return path_concat

    block2_output = block2(input_tensor=block2_input)
    block2_flattened = Flatten()(block2_output)

    dense1 = Dense(units=256, activation='relu')(block2_flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model