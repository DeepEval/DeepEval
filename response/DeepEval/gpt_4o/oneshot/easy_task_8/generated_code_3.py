import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise Separable Convolutional Layer
    separable_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.25)(separable_conv)

    # 1x1 Convolutional Layer
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv_1x1)

    # Flatten the output
    flatten_layer = Flatten()(dropout2)

    # Fully connected layer to produce classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model