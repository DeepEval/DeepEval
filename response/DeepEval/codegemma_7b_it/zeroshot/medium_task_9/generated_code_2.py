import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def basic_block(x, filters):
    # Main path
    conv1 = Conv2D(filters, (3, 3), padding='same')(x)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(filters, (3, 3), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)

    # Branch path
    conv_branch = Conv2D(filters, (3, 3), padding='same')(x)
    bn_branch = BatchNormalization()(conv_branch)

    # Feature fusion
    concat = concatenate([bn2, bn_branch], axis=-1)
    act_fusion = Activation('relu')(concat)

    return act_fusion

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(16, (3, 3), padding='same')(input_img)
    bn_init = BatchNormalization()(conv_init)
    act_init = Activation('relu')(bn_init)

    # Basic blocks
    block1 = basic_block(act_init, 16)
    block2 = basic_block(block1, 16)

    # Branch convolutional layer
    conv_branch = Conv2D(16, (3, 3), padding='same')(act_init)
    bn_branch = BatchNormalization()(conv_branch)

    # Feature fusion
    concat_branch = concatenate([block2, bn_branch], axis=-1)
    act_fusion = Activation('relu')(concat_branch)

    # Average pooling
    avg_pool = AveragePooling2D()(act_fusion)

    # Fully connected layer
    flatten = Flatten()(avg_pool)
    dense = Dense(10, activation='softmax')(flatten)

    # Model definition
    model = Model(inputs=input_img, outputs=dense)

    return model