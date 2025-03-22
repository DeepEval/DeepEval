import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, Activation

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_relu = Activation('relu')(block1_bn)

    # Block 2
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(block1_relu)
    block2_bn = BatchNormalization()(block2_conv)
    block2_relu = Activation('relu')(block2_bn)

    # Block 3
    block3_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(block2_relu)
    block3_bn = BatchNormalization()(block3_conv)
    block3_relu = Activation('relu')(block3_bn)

    # Outputs of each block
    output_block1 = block1_relu
    output_block2 = block2_relu
    output_block3 = block3_relu

    # Parallel branch processing the input directly
    parallel_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_relu = Activation('relu')(parallel_bn)

    # Merge all outputs by addition
    merged_output = Add()([output_block1, output_block2, output_block3, parallel_relu])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model