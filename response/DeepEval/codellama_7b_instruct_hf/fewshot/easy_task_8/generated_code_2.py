import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Create the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)

    # 1x1 convolutional layer for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)

    # Flatten the output and pass through a fully connected layer
    flatten_layer = Flatten()(conv2)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model