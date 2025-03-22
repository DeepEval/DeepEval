import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout
from keras.layers import Conv2D, Add, Lambda

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        add1 = Add()([conv1, input_tensor])
        path1 = Dropout(0.2)(add1)

        conv21 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv22 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv21)
        add2 = Add()([conv22, conv21])
        path2 = Dropout(0.2)(add2)

        conv31 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv32 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv31)
        path3 = Dropout(0.2)(conv32)

        pool4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv41 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4)
        path4 = Dropout(0.2)(conv41)

        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 64))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def main():
    model = dl_model()
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    main()