import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Basic block definition
    def basic_block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        # Branch path
        branch = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)
        # Addition
        output_tensor = Add()([x, branch])
        output_tensor = ReLU()(output_tensor)
        return output_tensor

    # First level of the residual structure
    x = basic_block(input_tensor=input_layer, filters=16)

    # Second level of the residual structure with two blocks
    x = basic_block(input_tensor=x, filters=16)
    branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(x)
    x = Add()([x, branch])
    x = ReLU()(x)

    # Third level of the residual structure
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    x = Add()([x, global_branch])

    # Final processing
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model