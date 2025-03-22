import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer to adjust feature dimensionality
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic block for the first level
    def basic_block(input_tensor):
        # Main path
        path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path = BatchNormalization()(path)
        path = Activation('relu')(path)

        # Branch
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Add outputs from main path and branch
        output_tensor = Add()([path, branch])

        return output_tensor

    # Residual block for the second level
    def residual_block(input_tensor):
        # Main path
        path = basic_block(input_tensor)

        # Branch
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch = BatchNormalization()(branch)
        branch = Activation('relu')(branch)
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch)

        # Add outputs from main path and branch
        output_tensor = Add()([path, branch])

        return output_tensor

    # First level
    level1 = basic_block(conv)

    # Second level
    level2 = residual_block(level1)
    level2 = residual_block(level2)

    # Global branch
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    # Third level
    level3 = Add()([level2, global_branch])
    level3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(level3)
    level3 = Flatten()(level3)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(level3)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model