import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)

    # Second 1x1 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)

    # 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(0.25)(conv3)

    # 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout3)
    dropout4 = Dropout(0.25)(conv4)

    # Restore channel count to match input
    restore_channels = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout4)

    # Add the restored channels to the original input
    combined = keras.layers.add([restore_channels, input_layer])

    # Flatten the result
    flatten_layer = Flatten()(combined)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout5 = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout5)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model