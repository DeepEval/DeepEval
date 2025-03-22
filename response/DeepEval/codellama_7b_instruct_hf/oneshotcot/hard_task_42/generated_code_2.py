import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Concatenate()([path1, path2, path3])
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)
    block1 = Flatten()(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.2)(block1)

    # Block 2
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(block1)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = Concatenate()([path1, path2, path3, path4])
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    block2 = Flatten()(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.2)(block2)

    # Output layer
    output = Dense(units=10, activation='softmax')(block2)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model