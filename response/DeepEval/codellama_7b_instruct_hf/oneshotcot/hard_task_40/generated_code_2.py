import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(pool1)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(pool2)
    flatten = Flatten()(pool3)
    concat = Concatenate()([pool1, pool2, pool3, flatten])
    dense1 = Dense(units=128, activation='relu')(concat)

    # Second block
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(dense1)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(dense1)
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(dense1)
    block4 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(dense1)
    dropout1 = Dropout(0.2)(block1)
    dropout2 = Dropout(0.2)(block2)
    dropout3 = Dropout(0.2)(block3)
    dropout4 = Dropout(0.2)(block4)
    concat_block = Concatenate()([dropout1, dropout2, dropout3, dropout4])
    dense2 = Dense(units=128, activation='relu')(concat_block)
    dense3 = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=dense3)
    return model