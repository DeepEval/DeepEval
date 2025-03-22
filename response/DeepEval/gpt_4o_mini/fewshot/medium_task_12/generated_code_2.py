import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        activated = ReLU()(norm)
        return activated

    # First Block
    block1_output = block(input_layer, filters=32)

    # Second Block
    block2_output = block(block1_output, filters=64)
    concatenated_2 = Concatenate(axis=-1)([block1_output, block2_output])

    # Third Block
    block3_output = block(concatenated_2, filters=128)
    concatenated_3 = Concatenate(axis=-1)([concatenated_2, block3_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated_3)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=dense2)

    return model