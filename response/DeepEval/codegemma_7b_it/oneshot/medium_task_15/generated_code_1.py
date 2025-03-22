import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Multiply, concatenate

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with batch normalization and ReLU activation
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn = BatchNormalization()(conv)
    relu = Activation('relu')(bn)

    # Global average pooling and two fully connected layers
    avg_pool = GlobalAveragePooling2D()(relu)
    dense1 = Dense(units=64, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape and multiply with initial features
    dense2_reshaped = Reshape((1, 1, 64))(dense2)
    weighted_features = Multiply()([dense2_reshaped, conv])

    # Concatenate with input layer
    concatenated = concatenate([input_layer, weighted_features])

    # 1x1 convolution and average pooling for dimensionality reduction and downsampling
    conv_down = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    avg_pool_down = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_down)

    # Single fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(avg_pool_down)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model