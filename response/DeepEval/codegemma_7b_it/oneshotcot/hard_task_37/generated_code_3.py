import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    path1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_conv1)
    path1_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_conv2)
    path1_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1_conv3)

    # Branch 2
    path2_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(path2_conv1)
    path2_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(path2_conv2)
    path2_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path2_conv3)

    # Branch 3
    path3_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path3_conv1)
    path3_conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path3_conv2)
    path3_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path3_conv3)

    # Parallel branch
    path4_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path4_conv1)

    # Concatenation and addition
    concat_layer = Concatenate()([path1_max_pool, path2_max_pool, path3_max_pool, path4_max_pool])
    add_layer = Add()([path1_max_pool, path2_max_pool, path3_max_pool, path4_max_pool])

    # Block output
    block_output = add_layer

    # Subsequent layers
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model