import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_conv)

    # Combine main and branch paths
    combined = Concatenate()([pooling, branch_pooling])

    # Add a flatten layer and a fully connected layer
    flatten = Flatten()(combined)
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)

    return model