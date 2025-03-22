import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same', input_shape=(32, 32, 3))(input_layer)
    layer_norm = LayerNormalization()(depthwise_conv)
    pointwise_conv1 = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), activation='relu')(layer_norm)
    pointwise_conv2 = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), activation='relu')(pointwise_conv1)

    # Branch path (identity, directly connected to input)
    branch_path = input_layer

    # Addition of the main path and branch path
    added = Add()([pointwise_conv2, branch_path])

    # Flatten the output and pass it through two fully connected layers
    flattened = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()