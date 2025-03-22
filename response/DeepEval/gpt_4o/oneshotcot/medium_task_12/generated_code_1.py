import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    def block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    x1 = block(input_layer, filters=32)
    
    # Second block with concatenation
    x2_input = Concatenate()([input_layer, x1])
    x2 = block(x2_input, filters=64)
    
    # Third block with concatenation
    x3_input = Concatenate()([x1, x2])
    x3 = block(x3_input, filters=128)

    # Flatten and dense layers
    flatten_layer = Flatten()(x3)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model