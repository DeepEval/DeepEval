import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, Lambda, Concatenate
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Main path
        conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
        # Branch path
        branch = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        # Add paths
        output_tensor = Add()([conv2, branch])
        return output_tensor

    block1_output = block1(input_layer)
    block1_output = BatchNormalization()(block1_output)

    # Second block
    def block2(x):
        # Split the input into three groups
        split_1 = Lambda(lambda z: z[:, :16, :16, :])(x)
        split_2 = Lambda(lambda z: z[:, 16:, :16, :])(x)
        split_3 = Lambda(lambda z: z[:, :16, 16:, :])(x)
        split_4 = Lambda(lambda z: z[:, 16:, 16:, :])(x)

        # Depthwise separable convolutions
        dsc1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_1)
        dsc2 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_2)
        dsc3 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_3)
        dsc4 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_4)

        # Concatenate outputs
        output_tensor = Concatenate()([dsc1, dsc2, dsc3, dsc4])
        return output_tensor

    block2_output = block2(block1_output)
    block2_output = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(block2_output)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()