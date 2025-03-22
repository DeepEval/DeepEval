import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer with dropout
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(rate=0.2)(conv1)

    # Second 1x1 convolutional layer with dropout
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(dropout1)
    dropout2 = Dropout(rate=0.2)(conv2)

    # 3x1 convolutional layer with dropout
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(dropout2)
    dropout3 = Dropout(rate=0.2)(conv3)

    # 1x3 convolutional layer with dropout
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(dropout3)
    dropout4 = Dropout(rate=0.2)(conv4)

    # Restore the number of channels to match the input's channel count with a 1x1 convolutional layer
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu', padding='same')(dropout4)

    # Combine the processed features with the original input via addition
    added = Add()([input_layer, conv5])

    # Flatten the result
    flatten_layer = Flatten()(added)

    # Fully connected layer to produce the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model