import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU)(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm

    def concatenate_blocks(input_tensor):
        concatenated = Concatenate(axis=-1)([block(input_tensor) for _ in range(3)])  # 3 blocks for demonstration
        return concatenated

    block1 = block(input_layer)
    block2 = block(block1)
    block3 = block(block2)
    concatenated = concatenate_blocks(block3)

    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and return the model
model = dl_model()
model.summary()