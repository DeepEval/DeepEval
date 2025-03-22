import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, DepthwiseConv2D, LayerNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same', depth_multiplier=1)(input_layer)
    main_path = LayerNormalization()(main_path)
    main_path = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1))(main_path)
    main_path = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1))(main_path)

    # Branch path
    branch_path = input_layer

    # Combine outputs of both paths
    combined_path = keras.layers.add([main_path, branch_path])

    # Flatten the output
    flatten_layer = Flatten()(combined_path)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()