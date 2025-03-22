import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path = Conv2D(filters=3, kernel_size=(7, 7), padding='same', activation='relu', depthwise=True)(input_layer)
    main_path = LayerNormalization()(main_path)

    # Two sequential 1x1 pointwise convolutions
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer  # Directly connects to the input

    # Combine the main path and branch path
    combined = Add()([main_path, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model