import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense, Reshape, BatchNormalization, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = Flatten()(path1)
        path1 = Dropout(0.2)(path1)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2 = Flatten()(path2)
        path2 = Dropout(0.3)(path2)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3 = Flatten()(path3)
        path3 = Dropout(0.4)(path3)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block1_output = block1(input_tensor=max_pooling)
    reshape_layer = Reshape((block1_output.shape[1], block1_output.shape[2], 1))(block1_output)

    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = BatchNormalization()(path1)
        path1 = Activation('relu')(path1)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = BatchNormalization()(path2)
        path2 = Activation('relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(path2)
        path2 = BatchNormalization()(path2)
        path2 = Activation('relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path3 = BatchNormalization()(path3)
        path3 = Activation('relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(path3)
        path3 = BatchNormalization()(path3)
        path3 = Activation('relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(path3)
        path3 = BatchNormalization()(path3)
        path3 = Activation('relu')(path3)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(path4)
        path4 = BatchNormalization()(path4)
        path4 = Activation('relu')(path4)

        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block2_output = block2(input_tensor=reshape_layer)
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model