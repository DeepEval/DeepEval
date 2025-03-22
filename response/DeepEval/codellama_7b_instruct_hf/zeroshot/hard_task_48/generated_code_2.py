from keras.layers import Input, Lambda, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block_1 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    block_1 = Concatenate()([
        Conv2D(32, (1, 1), activation='relu', padding='same')(block_1[0]),
        Conv2D(32, (3, 3), activation='relu', padding='same')(block_1[1]),
        Conv2D(32, (5, 5), activation='relu', padding='same')(block_1[2])
    ])
    block_1 = BatchNormalization()(block_1)

    # Block 2
    block_2 = Lambda(lambda x: tf.split(x, 4, axis=3))(block_1)
    block_2 = Concatenate()([
        Conv2D(64, (1, 1), activation='relu', padding='same')(block_2[0]),
        MaxPooling2D((2, 2))(Conv2D(64, (3, 3), activation='relu', padding='same')(block_2[1])),
        Conv2D(64, (1, 3), activation='relu', padding='same')(Conv2D(64, (3, 1), activation='relu', padding='same')(block_2[2])),
        Conv2D(64, (3, 1), activation='relu', padding='same')(Conv2D(64, (1, 3), activation='relu', padding='same')(block_2[3]))
    ])
    block_2 = BatchNormalization()(block_2)

    # Final layer
    output_layer = GlobalAveragePooling2D()(block_2)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model