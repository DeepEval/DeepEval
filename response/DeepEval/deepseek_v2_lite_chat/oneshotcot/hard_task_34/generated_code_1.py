import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Pathway 1
    input_layer = Input(shape=(28, 28, 1))
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)

    # Repeat block three times
    for _ in range(3):
        conv_layer = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_layer)
        pool_layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_layer)

    # Pathway 2
    branch_input = input_layer
    branch_conv_layer = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_input)
    branch_conv_layer.set_shape(input_shape)

    # Concatenate both pathways
    merged = Concatenate()([pool_layer, branch_conv_layer])

    # Add batch normalization and flatten layers
    batch_norm = BatchNormalization()(merged)
    flat = Flatten()(batch_norm)

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flat)
    fc2 = Dense(units=64, activation='relu')(fc1)
    fc3 = Dense(units=32, activation='relu')(fc2)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc3)

    # Model construction
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model