import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Convolutional layers with 3x3 kernel
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    # Branch 2: Convolutional layers with 5x5 kernel
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)

    # Combine branch outputs
    adding_layer = Add()([main_path, branch_path])

    # Global average pooling
    gap = GlobalAveragePooling2D()(adding_layer)

    # Fully connected layers for attention weights
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Attention weights
    dense3 = Dense(units=2, activation='softmax')(dense2)

    # Weighted output
    weighted_output = Multiply()([adding_layer, dense3])
    final_output = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model