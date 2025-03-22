import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # First block
    block1_output = block(input_layer, filters=32)

    # Second block
    block2_output = block(block1_output, filters=64)
    concatenated_1 = Concatenate(axis=-1)([block1_output, block2_output])

    # Third block
    block3_output = block(concatenated_1, filters=128)
    concatenated_2 = Concatenate(axis=-1)([concatenated_1, block3_output])

    # Flatten the concatenated output and add fully connected layers
    flatten = Flatten()(concatenated_2)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model