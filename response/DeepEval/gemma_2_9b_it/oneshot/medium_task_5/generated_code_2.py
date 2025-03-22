import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    main_path = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2))(y)

    # Combine Paths
    combined_output = Concatenate()([main_path, branch_path])

    # Flatten and Dense Layers
    x = Flatten()(combined_output)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model