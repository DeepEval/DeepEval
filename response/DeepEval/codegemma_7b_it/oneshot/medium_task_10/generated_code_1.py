import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Add

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Basic block
    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = BatchNormalization()(main_path)
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch = BatchNormalization()(branch)
        output_tensor = Add()([main_path, branch])
        output_tensor = Activation('relu')(output_tensor)

        return output_tensor

    # Second level residual blocks
    residual_block1 = basic_block(input_tensor=conv)
    residual_block2 = basic_block(input_tensor=residual_block1)

    # Third level residual block
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    global_branch = BatchNormalization()(global_branch)
    global_branch = Add()([global_branch, residual_block2])
    global_branch = Activation('relu')(global_branch)

    # Global average pooling
    avg_pool = AveragePooling2D()(global_branch)

    # Fully connected layer
    flatten_layer = Flatten()(avg_pool)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model