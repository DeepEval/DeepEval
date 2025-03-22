from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense, ReLU
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path - Block 1
    main_path = ReLU()(input_layer)
    main_path = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)

    # Main path - Block 2
    main_path = ReLU()(main_path)
    main_path = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)

    # Branch path
    branch_path = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Combine main and branch paths
    combined = Add()([main_path, branch_path])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model