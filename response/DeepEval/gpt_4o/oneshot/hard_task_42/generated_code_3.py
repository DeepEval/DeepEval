import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1_block1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2_block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3_block1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    path1_block1_flat = Flatten()(path1_block1)
    path2_block1_flat = Flatten()(path2_block1)
    path3_block1_flat = Flatten()(path3_block1)

    path1_block1_drop = Dropout(rate=0.5)(path1_block1_flat)
    path2_block1_drop = Dropout(rate=0.5)(path2_block1_flat)
    path3_block1_drop = Dropout(rate=0.5)(path3_block1_flat)

    block1_output = Concatenate()([path1_block1_drop, path2_block1_drop, path3_block1_drop])

    # Fully connected layer and reshape to 4D tensor
    fc = Dense(units=7*7*32, activation='relu')(block1_output)
    reshaped = Reshape((7, 7, 32))(fc)

    # Block 2
    path1_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)

    path2_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_block2)
    path2_block2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_block2)

    path3_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3_block2)

    path4_block2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    path4_block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4_block2)

    block2_output = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])

    # Fully connected layers for final classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model