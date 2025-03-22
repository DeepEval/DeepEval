import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel

    # First 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.3)(conv1)

    # Second 1x1 convolution
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(conv2)

    # 3x1 convolution
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(0.3)(conv3)

    # 1x3 convolution
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout3)
    dropout4 = Dropout(0.3)(conv4)

    # Final 1x1 convolution to restore channel count
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout4)

    # Add the original input to the processed features
    added_features = Add()([input_layer, conv5])

    # Flattening the output
    flatten_layer = Flatten()(added_features)

    # Fully connected layer to produce the final probability distribution
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model