import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, AveragePooling2D, Concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(branch2)
    branch3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch3)
    main_path_output = Concatenate()([branch1, branch2, branch3])
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_output)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    output = Add()([main_path_output, branch_path])

    # Classification
    flatten_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model