import keras
from keras.layers import Input, Conv2D, Activation, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model(): 
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = input_layer
    for _ in range(3):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)  
        # Feature concatenation for main path

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Feature Fusion
    x = Concatenate()([x, branch_path])
    x = x

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model