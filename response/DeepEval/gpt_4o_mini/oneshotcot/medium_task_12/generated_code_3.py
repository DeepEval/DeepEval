import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def conv_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # First block
    block1_output = conv_block(input_layer)

    # Second block
    block2_output = conv_block(block1_output)
    concatenated1 = Concatenate()([block1_output, block2_output])

    # Third block
    block3_output = conv_block(concatenated1)
    concatenated2 = Concatenate()([concatenated1, block3_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# To create the model, you can call the function:
model = dl_model()