import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    main_path_output = Add()([conv1, conv2])
    bath_norm1 = BatchNormalization()(main_path_output)

    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    flatten1 = Flatten()(pooling1)
    flatten2 = Flatten()(pooling2)
    flatten3 = Flatten()(pooling3)

    concat_output = Concatenate()([flatten1, flatten2, flatten3])

    dense1 = Dense(units=128, activation='relu')(concat_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model