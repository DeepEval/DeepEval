import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Multiply, Reshape, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Global Average Pooling to generate weights
    gap = GlobalAveragePooling2D()(input_layer)
    dense1_block1 = Dense(units=32, activation='relu')(gap)
    dense2_block1 = Dense(units=32 * 32 * 3, activation='sigmoid')(dense1_block1)
    reshape_weights = Reshape((32, 32, 3))(dense2_block1)
    weighted_output = Multiply()([input_layer, reshape_weights])

    # Block 2: Two 3x3 Convolutional layers followed by MaxPooling
    conv1_block2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(weighted_output)
    conv2_block2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1_block2)
    max_pool_block2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2_block2)

    # Branch from Block 1 to Block 2
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(weighted_output)

    # Fusing Block 2 output and the branch
    fused_output = Add()([max_pool_block2, branch])

    # Fully Connected Layers for Classification
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model