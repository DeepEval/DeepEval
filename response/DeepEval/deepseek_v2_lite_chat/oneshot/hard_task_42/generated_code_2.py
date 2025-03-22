import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.layers import Lambda

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    ### Block 1 ###
    ### Three parallel paths through max pooling layers ###
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path1)

    path2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(path2)

    path3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(path3)

    ### Flatten and apply dropout ###
    flat_path1 = Flatten()(path1)
    flat_path2 = Flatten()(path2)
    flat_path3 = Flatten()(path3)

    drop1 = Dropout(0.5)(flat_path1)
    drop2 = Dropout(0.5)(flat_path2)
    drop3 = Dropout(0.5)(flat_path3)

    ### Concatenate the flattened paths ###
    concat = Concatenate()( [drop1, drop2, drop3] )

    ### Block 2 ###
    ### Four parallel paths ###
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path7 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    path8 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path9 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path10 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path11 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    path12 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path13 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path5)
    path14 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path6)
    path15 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path7)

    ### Concatenate the outputs of the paths ###
    concat_block2 = Concatenate(axis=-1)([path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15])

    ### Fully connected layers for output ###
    dense1 = Dense(units=128, activation='relu')(concat_block2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model