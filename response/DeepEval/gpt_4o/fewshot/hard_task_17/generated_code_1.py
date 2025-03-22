import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    gap = GlobalAveragePooling2D()(input_layer)
    dense1_block1 = Dense(units=32*32*3, activation='relu')(gap)
    dense2_block1 = Dense(units=32*32*3, activation='sigmoid')(dense1_block1)
    weights = Reshape(target_shape=(32, 32, 3))(dense2_block1)
    weighted_features = Multiply()([input_layer, weights])

    # Block 2
    conv1_block2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2_block2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_block2)
    max_pooling_block2 = MaxPooling2D(pool_size=(2, 2))(conv2_block2)

    # Fusion
    fused_output = Add()([weighted_features, max_pooling_block2])

    # Classification
    flatten = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model