import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv_main_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool_main_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_main_1)
    conv_main_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool_main_1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_main_2)

    # Branch Path
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_branch)

    # Combine Paths
    combined = Add()([main_path, branch_path])

    # Fully Connected Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model