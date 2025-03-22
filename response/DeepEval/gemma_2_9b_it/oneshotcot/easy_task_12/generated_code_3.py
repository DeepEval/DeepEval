import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Conv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Path
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    branch = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Concatenate Paths
    x = Add()([x, branch]) 

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model