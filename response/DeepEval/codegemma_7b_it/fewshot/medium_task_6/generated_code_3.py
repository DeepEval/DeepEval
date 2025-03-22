import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Add

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv_init = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Parallel blocks
    block_outputs = []
    for i in range(3):
        # Convolution, batch normalization, ReLU activation
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_init if i == 0 else block_outputs[i - 1])
        batch_norm = BatchNormalization()(conv)
        relu = Activation('relu')(batch_norm)
        block_outputs.append(relu)

    # Add outputs of parallel blocks to initial convolution
    added_output = Add()([conv_init] + block_outputs)

    # Max pooling
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(added_output)

    # Flatten and fully connected layers
    flatten = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model