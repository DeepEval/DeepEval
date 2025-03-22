import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3)) 

    # Path 1
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    # Path 2
    y = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)

    # Combine paths
    z = Add()([x, y])

    # Flatten
    z = Flatten()(z)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(z)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model