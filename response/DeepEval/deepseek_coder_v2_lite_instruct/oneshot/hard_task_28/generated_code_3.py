import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(input_layer)
    main_path = LayerNormalization()(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine outputs of both paths
    combined_output = Add()([main_path, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model