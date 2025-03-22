import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def basic_block(input_tensor):
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)
        
        # Branch path
        branch = input_tensor
        
        # Addition
        output_tensor = Add()([relu, branch])
        return output_tensor

    # Initial convolutional layer to reduce dimensionality
    conv_init = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_init = BatchNormalization()(conv_init)
    relu_init = ReLU()(bn_init)

    # First block
    block1 = basic_block(relu_init)
    block1_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(block1)
    bn_block1 = BatchNormalization()(block1_conv)
    relu_block1 = ReLU()(bn_block1)

    # Second block
    block2 = basic_block(relu_block1)
    block2_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2)
    bn_block2 = BatchNormalization()(block2_conv)
    relu_block2 = ReLU()(bn_block2)

    # Addition of outputs from both blocks
    addition = Add()([relu_block2, relu_block1])

    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(addition)

    # Flatten layer
    flatten = Flatten()(avg_pool)

    # Fully connected layer
    fc = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=fc)

    return model