import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, ReLU, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Define the main path block
    def main_block(input_tensor):
        x = ReLU()(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None)(x)
        x = Concatenate(axis=-1)([input_tensor, x])
        return x
    
    # Main path
    x = main_block(input_layer)
    x = main_block(x)
    x = main_block(x)

    # Branch path
    branch_output = Conv2D(filters=x.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main path and branch path
    fused_output = Add()([x, branch_output])

    # Flatten and output layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model