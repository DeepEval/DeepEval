import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_bn1 = BatchNormalization()(block1_conv1)
    block1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_bn1)
    block1_bn2 = BatchNormalization()(block1_conv2)
    
    # Block 2
    block2_input = Concatenate()([block1_bn2, block1_conv2])
    block2_conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_input)
    block2_bn1 = BatchNormalization()(block2_conv1)
    block2_conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_bn1)
    block2_bn2 = BatchNormalization()(block2_conv2)

    # Block 3
    block3_input = Concatenate()([block2_bn2, block2_conv2])
    block3_conv1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3_input)
    block3_bn1 = BatchNormalization()(block3_conv1)
    block3_conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3_bn1)
    block3_bn2 = BatchNormalization()(block3_conv2)

    flatten_layer = Flatten()(block3_bn2)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model