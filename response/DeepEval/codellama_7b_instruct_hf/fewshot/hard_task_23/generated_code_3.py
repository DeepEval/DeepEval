import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Conv2DTranspose, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch: local feature extraction
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)

    # Second branch: downsampling
    branch_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)

    # Third branch: upsampling
    branch_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch_3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3)
    branch_3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3)

    # Concatenate branches and refine
    concat_layer = Concatenate()([branch_1, branch_2, branch_3])
    concat_layer = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Flatten and feed into fully connected layer
    flatten_layer = Flatten()(concat_layer)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define and return model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    return model