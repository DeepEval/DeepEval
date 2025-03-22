import keras
from keras.layers import Input, Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 

    # Path 1
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    path1_output = AveragePooling2D(pool_size=(2, 2))(x)

    # Path 2
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    path2_output = x

    # Combination
    x = Add()([path1_output, path2_output])

    # Flatten and Output
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model