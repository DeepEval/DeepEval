import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def main_path_block(input_tensor):
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        x = ReLU()(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x

    # First block of the main path
    x = main_path_block(input_layer)
    # Second block of the main path
    x = main_path_block(x)

    def branch_path_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        return x

    branch = branch_path_block(input_layer)

    # Sum the outputs of the main path and the branch path
    added = Add()([x, branch])

    # Flatten the result
    flattened = Flatten()(added)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()