import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Activation

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = Activation('relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate Outputs
    merged = Concatenate()([x, branch])

    # Flatten and Fully Connected Layer
    x = Flatten()(merged)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    return model