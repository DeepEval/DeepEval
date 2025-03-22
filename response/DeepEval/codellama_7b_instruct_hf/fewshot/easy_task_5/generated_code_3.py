import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolution layer to reduce input dimensionality
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # 3x3 convolutional layer to extract features
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)

    # 1x1 convolutional layer to restore feature map dimensionality
    conv3 = Conv2D(32, (1, 1), activation='relu')(conv2)

    # Flatten the output
    flatten = Flatten()(conv3)

    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model