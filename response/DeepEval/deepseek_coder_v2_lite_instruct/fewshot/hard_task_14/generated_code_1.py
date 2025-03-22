import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Flatten, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    gap = GlobalAveragePooling2D()(main_path)
    weights = Dense(units=32, activation='relu')(gap)
    weights = Dense(units=32, activation='sigmoid')(weights)  # Adjusting weights to match input shape
    weights = weights.reshape((1, 1, 32))  # Reshape to match input shape
    main_path_weighted = Multiply()([main_path, weights])

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combining Paths
    combined = Add()([main_path_weighted, branch_path])

    # Final Classification
    flatten = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model