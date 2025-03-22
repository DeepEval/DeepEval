import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Define the four parallel paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)

    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)

    path4 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)

    # Concatenate the outputs of all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Dense layer with 128 units
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer with softmax activation for 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model