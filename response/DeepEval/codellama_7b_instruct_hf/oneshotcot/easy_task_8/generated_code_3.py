import keras
from keras.layers import Input, DepthwiseSeparableConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    depthwise_separable_conv = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)

    # 1x1 convolutional layer for feature extraction
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(depthwise_separable_conv)

    # Dropout layer to mitigate overfitting
    dropout = Dropout(rate=0.2)(conv)

    # Flatten the output
    flatten = Flatten()(dropout)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model