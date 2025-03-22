import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)

    # Second 1x1 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout2 = Dropout(0.2)(conv2)

    # 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(conv2)
    dropout3 = Dropout(0.2)(conv3)

    # 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(conv1)
    dropout4 = Dropout(0.2)(conv4)

    # Add the processed features with the original input
    added = Add()([dropout3, dropout4, input_layer])

    # Flatten the result
    flatten_layer = Flatten()(added)

    # Fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=dense_layer)

    return model