import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)

    # Path 2
    x2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer) 

    # Concatenate outputs from both paths
    merged = Add()([x1, x2])

    # Flatten and classify
    x = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model