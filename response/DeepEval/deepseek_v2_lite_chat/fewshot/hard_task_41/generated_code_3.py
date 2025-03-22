import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, Lambda

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        flatten = Flatten()(Concatenate()([maxpool1, maxpool2, maxpool3]))
        dropout = Dropout(0.5)(flatten)
        return dropout

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concat = Concatenate()([conv1, conv2, conv3, conv4])
        dropout = Dropout(0.5)(concat)
        return dropout

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()