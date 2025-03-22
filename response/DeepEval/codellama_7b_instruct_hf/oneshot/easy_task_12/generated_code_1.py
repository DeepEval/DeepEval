import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_path)

    # Merge main and branch paths
    merged_path = Concatenate()([main_path, branch_path])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(merged_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model