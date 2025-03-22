import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1_1 = Dropout(0.25)(conv1_1)

    # Second 1x1 convolutional layer
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.25)(conv1_2)

    # 3x1 convolutional layer
    conv2 = Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout1_2)
    dropout2 = Dropout(0.25)(conv2)

    # 1x3 convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(0.25)(conv3)

    # Restoring channels to match the input
    conv4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout3)

    # Adding the processed features with the original input
    added = Add()([conv4, input_layer])

    # Flattening the result
    flattened = Flatten()(added)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model