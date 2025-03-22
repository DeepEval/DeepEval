import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_7 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(conv1_1)
    conv7_1 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(conv1_7)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv7_1)

    # Branch path
    branch_path = input_layer

    # Merge main path and branch path
    added = Add()([main_path, branch_path])

    # Flatten the output
    flatten = Flatten()(added)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model