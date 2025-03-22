import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def main_path(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    # Branch Path
    def branch_path(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return x

    # Concatenation and Fusion
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    concat_output = Concatenate(axis=-1)([main_output, branch_output])

    # Fully Connected Layer
    flatten_output = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model