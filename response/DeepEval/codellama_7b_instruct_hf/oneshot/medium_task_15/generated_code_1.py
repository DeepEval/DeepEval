import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Reshape, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Flatten and multiply
    flatten = Flatten()(dense2)
    multiply = Multiply()([flatten, input_layer])

    # Reshape and concatenate
    reshape = Reshape(target_shape=(32, 32, 1))(multiply)
    concatenate = Concatenate()([reshape, input_layer])

    # 1x1 convolution and average pooling
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenate)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Final fully connected layer
    final_dense = Dense(units=10, activation='softmax')(avg_pool)

    model = keras.Model(inputs=input_layer, outputs=final_dense)

    return model