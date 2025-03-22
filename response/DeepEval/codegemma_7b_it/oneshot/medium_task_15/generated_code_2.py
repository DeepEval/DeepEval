import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, concatenate, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Extract initial features
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = Activation('relu')(batch_norm)

    # Compress feature maps
    global_avg_pool = GlobalAveragePooling2D()(relu)
    flatten = Flatten()(global_avg_pool)

    # Adjust dimensions to match initial features
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape to match initial feature size
    dense2_reshaped = Reshape((1, 1, 64))(dense2)

    # Generate weighted feature maps
    multiply = Multiply()([dense2_reshaped, relu])

    # Concatenate with input layer
    concat = concatenate([multiply, input_layer])

    # Reduce dimensionality and downsample
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')(concat)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Single fully connected layer for classification
    flatten2 = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model