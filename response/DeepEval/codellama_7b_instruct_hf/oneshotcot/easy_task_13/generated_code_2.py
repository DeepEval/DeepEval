import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layers
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    conv2 = Conv2D(32, (1, 1), activation='relu')(conv1)

    # 3x1 convolutional layers
    conv3 = Conv2D(32, (3, 1), activation='relu')(conv2)

    # 1x3 convolutional layers
    conv4 = Conv2D(32, (1, 3), activation='relu')(conv3)

    # restore number of channels
    conv5 = Conv2D(1, (1, 1), activation='relu')(conv4)

    # dropout layers
    dropout1 = Dropout(0.2)(conv1)
    dropout2 = Dropout(0.2)(conv2)
    dropout3 = Dropout(0.2)(conv3)
    dropout4 = Dropout(0.2)(conv4)

    # combine features
    concat = Concatenate()([conv5, dropout1, dropout2, dropout3, dropout4])

    # flatten layer
    flatten = Flatten()(concat)

    # fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model