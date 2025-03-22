import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Level 1: Basic Block
    def basic_block(x):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        shortcut = x  # Direct connection from input to output
        output = Add()([relu, shortcut])  # Element-wise addition
        return output

    level1 = basic_block(input_layer)

    # Level 2: Two Basic Blocks with branches
    level2_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(level1)
    level2_main = basic_block(level1)
    level2_main = basic_block(level2_main)
    level2 = Add()([level2_main, level2_branch])  # Addition of main path and branch

    # Level 3: One Basic Block with a global branch
    global_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(level2)
    level3 = Add()([global_branch, level2])  # Addition of global branch and level 2 output

    # Final processing
    avg_pool = AveragePooling2D(pool_size=(4, 4))(level3)
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model