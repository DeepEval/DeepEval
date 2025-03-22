import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer for the MNIST dataset
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        # First block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Second block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        return x

    # Create main path output
    main_output = main_path(input_layer)

    # Branch path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(main_output)

    # Sum the outputs of the main path and branch path
    summed_output = Add()([main_output, branch_output])

    # Flatten the result
    flatten_layer = Flatten()(summed_output)

    # Fully connected layer for output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model