import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolution
    conv1_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1_3x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_3x3)

    # Branch 2: 5x5 convolution
    conv1_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1_5x5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_5x5)

    # Concatenate outputs of both branches
    concat = Concatenate()([max_pooling1_3x3, max_pooling1_5x5])

    # Global average pooling
    gap = Lambda(lambda x: keras.backend.mean(x, axis=(1, 2)))(concat)

    # Two fully connected layers for attention weights
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Attention weights
    attention_weights = Dense(units=1, activation='softmax')(dense2)

    # Weighted output
    weighted_output = Lambda(lambda x: x[0] * x[1])([concat, attention_weights])

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model