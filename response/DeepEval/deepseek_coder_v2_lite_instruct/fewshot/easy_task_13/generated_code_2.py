import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout1_1 = Dropout(0.25)(conv1_1)

    # Second 1x1 convolutional layer
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.25)(conv1_2)

    # 3x1 convolutional layer
    conv2 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(dropout1_2)
    dropout2 = Dropout(0.25)(conv2)

    # 1x3 convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(1, 3), activation='relu')(dropout2)
    dropout3 = Dropout(0.25)(conv3)

    # Restore the number of channels to match the input
    conv4 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(dropout3)

    # Add the processed features with the original input
    added = Add()([conv4, input_layer])

    # Flatten the result
    flatten_layer = Flatten()(added)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model