import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, TransposeConv2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    branch_output = conv1

    # Main Path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_output)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    main_path = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch Path
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_output)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_path)
    branch_path = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_path)
    branch_path = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_path)

    # Concatenate and Fuse
    main_path = Concatenate()([main_path, branch_path])
    main_path = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Output
    flatten_layer = Flatten()(main_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model