import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block1():
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        add = Add()([conv1, conv2])
        branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(add)
        return add, branch

    def block2():
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
        concat = Concatenate()([maxpool1, maxpool2, maxpool3])
        flatten = Flatten()(concat)
        return flatten

    add1, branch = block1()
    flatten = block2()
    concat_output = Add()([add1, branch])
    dense1 = Dense(units=128, activation='relu')(concat_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model