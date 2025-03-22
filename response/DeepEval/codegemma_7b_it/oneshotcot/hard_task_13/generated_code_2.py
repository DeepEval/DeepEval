import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Second block
    pool = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=64, activation='relu')(pool)
    dense2 = Dense(units=block1_output.shape[3], activation='sigmoid')(dense1)
    dense2 = Reshape((block1_output.shape[1], block1_output.shape[2], block1_output.shape[3]))(dense2)
    block2_output = multiply([dense2, block1_output])

    # Final layer
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model