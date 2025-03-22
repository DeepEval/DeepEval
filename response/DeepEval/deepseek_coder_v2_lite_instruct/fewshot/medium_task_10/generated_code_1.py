import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def basic_block(input_tensor, filters):
        # Main path
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Branch path (direct connection)
        branch = input_tensor
        
        # Addition
        output_tensor = Add()([x, branch])
        output_tensor = ReLU()(output_tensor)
        return output_tensor

    # Level 1: Basic block
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3))(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    level1 = basic_block(x, filters=16)

    # Level 2: Two residual blocks
    x = basic_block(level1, filters=16)
    x = basic_block(x, filters=16)
    level2 = Add()([level1, x])

    # Level 3: Global branch capturing initial convolution output
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    x = Add()([level2, global_branch])

    # Final processing
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model