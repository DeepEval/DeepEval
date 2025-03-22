import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    combined_output = Concatenate()([main_path, branch_path])
    batch_norm = BatchNormalization()(combined_output)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Second block
    max_pool_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    max_pool_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flattened_output = Concatenate()([max_pool_1, max_pool_2, max_pool_3])
    flattened_output = Flatten()(flattened_output)
    dense3 = Dense(units=128, activation='relu')(flattened_output)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model