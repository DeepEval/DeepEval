import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, LayerNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Depthwise Separable Convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    norm_depthwise_conv = LayerNormalization(epsilon=1e-6)(depthwise_conv)  # Apply layer normalization

    # Fully Connected Layers
    fc1 = Dense(units=32, activation='relu')(norm_depthwise_conv)  # First fully connected layer
    fc2 = Dense(units=32, activation='relu')(fc1)  # Second fully connected layer

    # Addition Operation
    addition = Add()([input_layer, fc2])

    # Flatten and Output Layer
    flatten = Flatten()(addition)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model