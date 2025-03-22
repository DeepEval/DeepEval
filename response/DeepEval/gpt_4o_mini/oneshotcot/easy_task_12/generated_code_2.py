import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path block 1
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Main path block 2
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)

    # Summing both paths
    combined = Add()([x, branch])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model