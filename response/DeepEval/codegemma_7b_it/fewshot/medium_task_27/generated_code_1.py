import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional branch 1
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)

    # Convolutional branch 2
    conv2_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool2)

    # Combined features through addition
    merged = Add()([conv1_2, conv2_2])

    # Global average pooling
    gap = GlobalAveragePooling2D()(merged)

    # Fully connected layers for attention weights
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Weighted output
    weighted_output = Multiply()([merged, dense2])

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model