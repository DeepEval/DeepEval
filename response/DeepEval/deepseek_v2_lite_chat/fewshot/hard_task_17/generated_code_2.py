import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=128)(x)

    # Block 2
    branch_input = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    branch_input = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(branch_input)
    branch_input = GlobalAveragePooling2D()(branch_input)
    branch_input = Dense(units=64)(branch_input)

    # Branch connection
    x = Add()([x, branch_input])

    # Final classification
    x = Dense(units=10)(x)
    model = Model(inputs=input_layer, outputs=x)

    return model