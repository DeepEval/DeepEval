import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, GlobalAveragePooling2D, Reshape, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Second block
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    main_path = block(input_tensor=avg_pool)

    # Combine input and main path
    main_path = keras.layers.add([avg_pool, main_path])

    # Global average pooling and fully connected layers
    avg_pool = GlobalAveragePooling2D()(main_path)
    fc1 = Dense(units=16, activation='relu')(avg_pool)
    fc2 = Dense(units=16, activation='relu')(fc1)

    # Reshape and multiply
    fc2 = Reshape((1, 1, 16))(fc2)
    weights = Multiply()([fc2, main_path])

    # Flatten and classification
    flatten = Flatten()(weights)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model