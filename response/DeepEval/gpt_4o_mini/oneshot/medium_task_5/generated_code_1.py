import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2))(branch_path)

    # Combine both paths using addition
    combined_output = Add()([main_path, branch_path])

    # Flatten and Dense layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model