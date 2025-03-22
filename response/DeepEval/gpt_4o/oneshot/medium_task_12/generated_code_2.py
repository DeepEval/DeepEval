import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block(input_tensor, filters, kernel_size=(3, 3)):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        activation = ReLU()(batch_norm)
        return activation

    # First block
    block1_output = block(input_layer, filters=32)

    # Second block, concatenating with the output of the first block
    concat1 = Concatenate()([input_layer, block1_output])
    block2_output = block(concat1, filters=64)

    # Third block, concatenating with the output of the first and second blocks
    concat2 = Concatenate()([concat1, block2_output])
    block3_output = block(concat2, filters=128)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model