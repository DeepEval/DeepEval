import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization, ReLU
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    def main_path_block(input_tensor):
        # Separable convolution layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None)(input_tensor)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        return conv

    # Main path
    main_path_output = input_layer
    for _ in range(3):
        main_path_output = main_path_block(main_path_output)

    # Branch path
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of the two paths
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and final classification
    flatten_output = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model