import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv_initial = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = UpSampling2D(size=(2, 2))(branch_2)
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    branch_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3)
    branch_3 = UpSampling2D(size=(4, 4))(branch_3)
    concat_main = Concatenate()([branch_1, branch_2, branch_3])
    conv_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_main)

    # Branch Path
    conv_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Paths
    add_path = keras.layers.Add()([conv_main, conv_branch])
    bath_norm = BatchNormalization()(add_path)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model