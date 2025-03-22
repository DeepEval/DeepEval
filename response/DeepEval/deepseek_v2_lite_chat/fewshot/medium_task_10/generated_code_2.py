import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Adjusted input feature dimensionality to 16

    # Basic Block
    def basic_block(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU)(input_tensor)
        bn = BatchNormalization()(conv)
        return bn

    # First Level - Single Basic Block
    main_path = basic_block(input_tensor=input_layer)
    branch = basic_block(input_tensor=input_layer)
    added_output = Add()([main_path, branch])

    # Second Level - Two Basic Blocks
    def residual_block(input_tensor):
        main_path = basic_block(input_tensor=input_tensor)
        shortcut = basic_block(input_tensor=input_tensor)
        return ReLU(activation='relu')(Add()([main_path, shortcut]))

    second_level_output = residual_block(input_tensor=added_output)

    # Third Level - Global Branch
    def global_branch(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU)(input_tensor)
        added_global_output = Add()([second_level_output, conv])
        return added_global_output

    third_level_output = global_branch(input_tensor=input_layer)

    # Final Classification Layer
    flatten = Flatten()(third_level_output)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()