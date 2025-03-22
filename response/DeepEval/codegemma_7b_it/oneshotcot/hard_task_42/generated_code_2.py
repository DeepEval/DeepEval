import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model(): 

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(max_pooling)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(max_pooling)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(max_pooling)
    output_tensor = Concatenate()([path1, path2, path3])
    output_tensor = Dropout(0.25)(output_tensor)
    output_tensor = Flatten()(output_tensor)

    # Block 2
    reshape_layer = Reshape((1, 1, -1))(output_tensor)

    path1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=128, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=128, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path3 = Conv2D(filters=128, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=128, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=128, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(reshape_layer)
    path4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(path4)
    output_tensor = Concatenate()([path1, path2, path3, path4])
    output_tensor = Dropout(0.25)(output_tensor)
    output_tensor = Flatten()(output_tensor)

    # Output layers
    dense1 = Dense(units=1024, activation='relu')(output_tensor)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model