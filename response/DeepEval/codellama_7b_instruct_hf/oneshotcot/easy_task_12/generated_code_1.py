import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Sum the outputs from both paths
    output_tensor = Concatenate()([main_path, branch_path])

    # Flatten and fully connected layers
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(units=128, activation='relu')(output_tensor)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model