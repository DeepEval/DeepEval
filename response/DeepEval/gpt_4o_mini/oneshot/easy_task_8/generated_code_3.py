import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise Separable Convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Adding Dropout after Depthwise Convolution
    dropout1 = Dropout(rate=0.25)(depthwise_conv)

    # 1x1 Convolution
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    # Adding Dropout after 1x1 Convolution
    dropout2 = Dropout(rate=0.25)(conv_1x1)

    # Flattening the output
    flatten_layer = Flatten()(dropout2)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model