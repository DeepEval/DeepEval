import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First level: Basic block
    conv_first = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm_first = BatchNormalization()(conv_first)
    relu_first = Activation('relu')(batch_norm_first)

    # Second level: Residual blocks
    def residual_block(input_tensor):
        # Main path
        conv_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm_main = BatchNormalization()(conv_main)
        relu_main = Activation('relu')(batch_norm_main)

        conv_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu_main)
        batch_norm_main = BatchNormalization()(conv_main)

        # Branch
        conv_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        batch_norm_branch = BatchNormalization()(conv_branch)

        # Add outputs
        add_output = Add()([relu_main, batch_norm_branch])
        relu_output = Activation('relu')(add_output)

        return relu_output

    residual_block_output_1 = residual_block(input_tensor=relu_first)
    residual_block_output_2 = residual_block(input_tensor=residual_block_output_1)

    # Third level: Global branch
    conv_global = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm_global = BatchNormalization()(conv_global)
    relu_global = Activation('relu')(batch_norm_global)

    # Average pooling and fully connected layer
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=1)(relu_global)
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model