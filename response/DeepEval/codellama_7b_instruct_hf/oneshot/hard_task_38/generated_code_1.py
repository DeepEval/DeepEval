import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    block_output_1 = batch_norm(block(max_pooling, 3))

    # Pathway 2
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    block_output_2 = batch_norm(block(max_pooling, 3))

    # Merge pathways
    concatenated_output = Concatenate()([block_output_1, block_output_2])

    # Fully connected layers
    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def block(input_tensor, filters):
    # Batch normalization
    x = BatchNormalization()(input_tensor)
    # ReLU activation
    x = ReLU()(x)
    # 3x3 convolutional layer
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    return x

def batch_norm(x):
    # Batch normalization
    x = BatchNormalization()(x)
    return x