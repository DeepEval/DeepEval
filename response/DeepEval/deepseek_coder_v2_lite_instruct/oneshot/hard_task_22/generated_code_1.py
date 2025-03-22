import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    def main_path(x):
        # Split the input into three groups along the channel axis
        split_1 = Lambda(lambda z: z[:, :, :, :10])(x)
        split_2 = Lambda(lambda z: z[:, :, :, 10:20])(x)
        split_3 = Lambda(lambda z: z[:, :, :, 20:])(x)

        # Multi-scale feature extraction
        conv_1x1 = Conv2D(10, (1, 1), padding='same', activation='relu')(split_1)
        conv_3x3 = Conv2D(10, (3, 3), padding='same', activation='relu')(split_2)
        conv_5x5 = Conv2D(10, (5, 5), padding='same', activation='relu')(split_3)

        # Concatenate the outputs from these groups
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated

    main_output = main_path(input_layer)

    # Branch Path
    branch_output = Conv2D(10, (1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the outputs from both paths through addition
    fused_output = Add()([main_output, branch_output])

    # Flatten the result
    flattened = Flatten()(fused_output)

    # Pass through two fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense1)

    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()