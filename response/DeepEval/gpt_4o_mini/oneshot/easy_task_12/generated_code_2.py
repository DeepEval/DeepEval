import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Path Block 1
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Main Path Block 2
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)

    # Sum the outputs of the main and branch paths
    combined = Add()([x, branch])
    
    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model