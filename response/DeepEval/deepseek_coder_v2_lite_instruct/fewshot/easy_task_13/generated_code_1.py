import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1_1 = Dropout(0.25)(conv1_1)

    # Second 1x1 convolutional layer
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.25)(conv1_2)

    # 3x1 convolutional layer
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_2)
    dropout3_1 = Dropout(0.25)(conv3_1)

    # 1x3 convolutional layer
    conv1_3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout3_1)
    dropout1_3 = Dropout(0.25)(conv1_3)

    # Restore channel count to match input
    conv1_4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_3)

    # Add the processed features with the original input
    added = Add()([conv1_4, input_layer])

    # Flatten the result
    flattened = Flatten()(added)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model