import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(gap)
        dense2 = Dense(units=32, activation='relu')(dense1)
        reshaped = keras.layers.Reshape(target_shape=(32, 32, 3))(dense2)
        weighted_input = Multiply()([input_tensor, reshaped])
        return weighted_input

    block1_output = block_1(input_tensor=input_layer)

    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(block1_output)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    maxpool = MaxPooling2D()(conv2)

    # Branch from Block 1
    branch = block_1(input_tensor=maxpool)

    # Fused output
    fused_output = Add()([block1_output, branch])

    # Final classification
    flatten = keras.layers.Flatten()(fused_output)
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model