import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path: Two consecutive 3x3 convolutional layers followed by a max pooling layer
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)

    # Branch path: A single 5x5 convolutional layer
    branch_path = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of the main path and branch path
    combined = Concatenate()([main_path, branch_path])

    # Flatten the combined features
    flatten_layer = Flatten()(combined)

    # Fully connected layers to map to probability distribution across 10 classes
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model