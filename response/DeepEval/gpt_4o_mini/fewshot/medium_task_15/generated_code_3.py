import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)

    # Fully Connected Layers
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape to match the initial feature map size
    reshaped = Reshape((1, 1, 32))(dense2)

    # Weighted Feature Maps
    multiplied = Multiply()([relu, reshaped])

    # Concatenate with Input Layer
    concatenated = Concatenate()([input_layer, multiplied])

    # Reduce dimensionality with 1x1 Convolution
    conv_final = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Average Pooling
    avg_pool_final = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_final)

    # Final Fully Connected Layer
    flatten = keras.layers.Flatten()(avg_pool_final)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model