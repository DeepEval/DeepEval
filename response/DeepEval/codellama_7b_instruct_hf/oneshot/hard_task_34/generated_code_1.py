from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = input_layer
    for i in range(3):
        # Specific block
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Concatenate()([main_path, main_path])

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse features
    fused_features = Add()([main_path, branch_path])

    # Flatten and output
    flattened_features = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flattened_features)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model