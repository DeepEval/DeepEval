import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    block1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1)
    block1 = Concatenate()([block1, block1, block1, block1])

    block1_output = block1
    block1_output = BatchNormalization()(block1_output)
    block1_output = Flatten()(block1_output)

    block2 = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=128, activation='relu')(block2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()