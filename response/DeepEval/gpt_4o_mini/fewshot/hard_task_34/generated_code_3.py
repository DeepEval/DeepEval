import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path block definition
    def main_path_block(input_tensor):
        conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return Concatenate()([input_tensor, conv])

    # Creating the main path by stacking blocks
    main_path = input_layer
    for _ in range(3):
        main_path = main_path_block(main_path)

    # Branch path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Feature fusion
    fused_output = Add()([main_path, branch_path])

    # Flattening and output layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model