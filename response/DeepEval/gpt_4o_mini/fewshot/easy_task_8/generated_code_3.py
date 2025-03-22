import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolution layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.25)(depthwise_conv)  # Dropout after depthwise convolution

    # 1x1 convolution layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv1x1)  # Dropout after 1x1 convolution

    # Flatten the output
    flatten_layer = Flatten()(dropout2)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model