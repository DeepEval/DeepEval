import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, Conv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 1))(dense)

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(0.2)(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        dropout2 = Dropout(0.2)(conv3)

        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout3 = Dropout(0.2)(conv4)

        pool = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1)
        conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
        dropout4 = Dropout(0.2)(conv5)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3, dropout4])
        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model