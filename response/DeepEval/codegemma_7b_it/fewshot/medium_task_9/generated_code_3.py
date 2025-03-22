import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, Flatten, Dense

def residual_block(input_tensor, filters, kernel_size, strides):
    # Main path
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)

    # Branch path
    branch = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    branch_bn = BatchNormalization()(branch)
    branch_act = Activation('relu')(branch_bn)

    # Feature fusion
    output_tensor = Add()([act, branch_act])
    output_tensor = Activation('relu')(output_tensor)

    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_init = BatchNormalization()(conv_init)
    act_init = Activation('relu')(bn_init)

    # Basic blocks
    block1 = residual_block(input_tensor=act_init, filters=16, kernel_size=(3, 3), strides=1)
    block2 = residual_block(input_tensor=block1, filters=32, kernel_size=(3, 3), strides=2)

    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    branch_bn = BatchNormalization()(branch_conv)
    branch_act = Activation('relu')(branch_bn)

    # Feature fusion
    combined = Add()([block2, branch_act])

    # Average pooling
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(combined)

    # Flattening and fully connected layer
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model