import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Flatten, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Local feature extraction branch
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_init)
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_local)

    # Downsampling and upsampling branches
    branch_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_init)
    branch_pool_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_pool_1)
    branch_pool_1 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_pool_1)

    branch_pool_2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv_init)
    branch_pool_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_pool_2)
    branch_pool_2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu')(branch_pool_2)

    # Concatenation and refinement
    concat = Concatenate()([branch_local, branch_pool_1, branch_pool_2])
    branch_final = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Fully connected layer
    flatten = Flatten()(branch_final)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model