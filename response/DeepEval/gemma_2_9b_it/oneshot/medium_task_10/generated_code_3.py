import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Level 1: Initial Convolution and Basic Block
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        shortcut = input_tensor
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        return x

    x = basic_block(x)

    # Level 2: Two Residual Blocks
    def residual_block(input_tensor):
        shortcut = input_tensor
        x = basic_block(input_tensor)
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        x = Add()([x, branch])
        return x

    x = residual_block(x)
    x = residual_block(x)

    # Level 3: Global Branch and Feature Fusion
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    x = Add()([x, global_branch])

    # Average Pooling and Classification
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model