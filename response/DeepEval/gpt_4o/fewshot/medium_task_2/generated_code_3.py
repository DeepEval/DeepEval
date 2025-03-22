import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv2)

    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined = Concatenate()([main_path, branch_conv])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model