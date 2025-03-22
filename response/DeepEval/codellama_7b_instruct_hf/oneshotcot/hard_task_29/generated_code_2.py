import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv)

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main and branch paths through addition
    output_tensor = Concatenate()([main_conv2, branch_conv])

    # Max pooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(output_tensor)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output_tensor)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(output_tensor)

    # Flatten pooling layers and concatenate
    flattened_pool1 = Flatten()(pool1)
    flattened_pool2 = Flatten()(pool2)
    flattened_pool3 = Flatten()(pool3)
    concatenated_pool = Concatenate()([flattened_pool1, flattened_pool2, flattened_pool3])

    # Batch normalization and flatten layer
    bath_norm = BatchNormalization()(concatenated_pool)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model