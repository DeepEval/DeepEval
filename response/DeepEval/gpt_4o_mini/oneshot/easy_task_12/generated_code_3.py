import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        return x

    main_output = main_path(input_layer)

    # Branch path
    branch_output = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Ensuring both paths have the same dimensions before summing
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch_output)

    # Summing outputs from both paths
    combined_output = Add()([main_output, branch_output])

    # Flattening and fully connected layer
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model