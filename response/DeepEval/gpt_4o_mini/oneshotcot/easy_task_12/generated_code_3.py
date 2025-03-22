import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def main_path(input_tensor):
        # Block 1
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Block 2
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        return x
    
    main_output = main_path(input_layer)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(main_output)

    # Combine Main and Branch Paths
    combined_output = Add()([main_output, branch_path])

    # Flatten and Dense Layers
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model