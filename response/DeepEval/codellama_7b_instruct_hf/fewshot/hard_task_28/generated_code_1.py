import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Flatten

 和 return model
def dl_model():
    # Define the input layer with shape (32, 32, 3) for 3-channel RGB images
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path with a depthwise convolution layer followed by layer normalization
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    layer_normalization = LayerNormalization()(depthwise_conv)

    # Define the branch path with two sequential 1x1 pointwise convolution layers
    pointwise_conv1 = Dense(64, activation='relu')(layer_normalization)
    pointwise_conv2 = Dense(64, activation='relu')(pointwise_conv1)

    # Define the addition layer to combine the outputs of the main and branch paths
    addition_layer = Add()([pointwise_conv2, depthwise_conv])

    # Define the flatten layer to flatten the output of the addition layer into a one-dimensional vector
    flatten_layer = Flatten()(addition_layer)

    # Define the fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the output layer with the final classification
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model by defining the input and output layers
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model